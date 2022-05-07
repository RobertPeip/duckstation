#include "cpu_core.h"
#include "bus.h"
#include "common/align.h"
#include "common/file_system.h"
#include "common/log.h"
#include "common/state_wrapper.h"
#include "cpu_core_private.h"
#include "cpu_disasm.h"
#include "cpu_recompiler_thunks.h"
#include "gte.h"
#include "host_interface.h"
#include "pgxp.h"
#include "settings.h"
#include "system.h"
#include "timing_event.h"
#include "timers.h"
#include "interrupt_controller.h"
#include "gpu_sw_backend.h"
#include "gpu.h"
#include "dma.h"
#include "pad.h"
#include "mdec.h"
#include "cdrom.h"
#include "spu.h"
#include <cstdio>
#include <fstream>

Log_SetChannel(CPU::Core);

namespace CPU {

static void SetPC(u32 new_pc);
static void UpdateLoadDelay();
static void Branch(u32 target);
static void FlushPipeline(bool isException);

State g_state;
bool g_using_interpreter = false;
bool TRACE_EXECUTION = false;

static std::FILE* s_log_file = nullptr;
static bool s_log_file_opened = false;
static bool s_trace_to_log = false;

static constexpr u32 INVALID_BREAKPOINT_PC = UINT32_C(0xFFFFFFFF);
static std::vector<Breakpoint> s_breakpoints;
static u32 s_breakpoint_counter = 1;
static u32 s_last_breakpoint_check_pc = INVALID_BREAKPOINT_PC;
static bool s_single_step = false;

bool IsTraceEnabled()
{
  return s_trace_to_log;
}

void StartTrace()
{
  if (s_trace_to_log)
    return;

  s_trace_to_log = true;
  UpdateDebugDispatcherFlag();
}

void StopTrace()
{
  if (!s_trace_to_log)
    return;

  if (s_log_file)
    std::fclose(s_log_file);

  s_log_file_opened = false;
  s_trace_to_log = false;
  UpdateDebugDispatcherFlag();
}

void WriteToExecutionLog(const char* format, ...)
{
  std::va_list ap;
  va_start(ap, format);

  if (!s_log_file_opened)
  {
    s_log_file = FileSystem::OpenCFile("cpu_log.txt", "wb");
    s_log_file_opened = true;
  }

  if (s_log_file)
  {
    std::vfprintf(s_log_file, format, ap);
#ifdef _DEBUG
    std::fflush(s_log_file);
#endif
  }

  va_end(ap);
}

void Initialize()
{
  // From nocash spec.
  g_state.cop0_regs.PRID = UINT32_C(0x00000002);

  g_state.use_debug_dispatcher = false;
  s_breakpoints.clear();
  s_breakpoint_counter = 1;
  s_last_breakpoint_check_pc = INVALID_BREAKPOINT_PC;
  s_single_step = false;

  UpdateFastmemBase();

  GTE::Initialize();
}

void Shutdown()
{
  ClearBreakpoints();
  StopTrace();
}

void Reset()
{
  g_state.pending_ticks = 0;
  g_state.downcount = 0;

  g_state.regs = {};

  g_state.cop0_regs.BPC = 0;
  g_state.cop0_regs.BDA = 0;
  g_state.cop0_regs.TAR = 0;
  g_state.cop0_regs.BadVaddr = 0;
  g_state.cop0_regs.BDAM = 0;
  g_state.cop0_regs.BPCM = 0;
  g_state.cop0_regs.EPC = 0;
  g_state.cop0_regs.sr.bits = 0;
  g_state.cop0_regs.cause.bits = 0;

  ClearICache();
  UpdateFastmemBase();

  GTE::Reset();

  SetPC(RESET_VECTOR);
}

bool DoState(StateWrapper& sw)
{
  sw.Do(&g_state.pending_ticks);
  sw.Do(&g_state.downcount);
  sw.DoArray(g_state.regs.r, countof(g_state.regs.r));
  sw.Do(&g_state.cop0_regs.BPC);
  sw.Do(&g_state.cop0_regs.BDA);
  sw.Do(&g_state.cop0_regs.TAR);
  sw.Do(&g_state.cop0_regs.BadVaddr);
  sw.Do(&g_state.cop0_regs.BDAM);
  sw.Do(&g_state.cop0_regs.BPCM);
  sw.Do(&g_state.cop0_regs.EPC);
  sw.Do(&g_state.cop0_regs.PRID);
  sw.Do(&g_state.cop0_regs.sr.bits);
  sw.Do(&g_state.cop0_regs.cause.bits);
  sw.Do(&g_state.cop0_regs.dcic.bits);
  sw.Do(&g_state.next_instruction.bits);
  sw.Do(&g_state.current_instruction.bits);
  sw.Do(&g_state.current_instruction_pc);
  sw.Do(&g_state.current_instruction_in_branch_delay_slot);
  sw.Do(&g_state.current_instruction_was_branch_taken);
  sw.Do(&g_state.next_instruction_is_branch_delay_slot);
  sw.Do(&g_state.branch_was_taken);
  sw.Do(&g_state.exception_raised);
  sw.Do(&g_state.interrupt_delay);
  sw.Do(&g_state.load_delay_reg);
  sw.Do(&g_state.load_delay_value);
  sw.Do(&g_state.next_load_delay_reg);
  sw.Do(&g_state.next_load_delay_value);
  sw.Do(&g_state.cache_control.bits);
  sw.DoBytes(g_state.dcache.data(), g_state.dcache.size());

  if (!GTE::DoState(sw))
    return false;

  if (sw.GetVersion() < 48)
  {
    DebugAssert(sw.IsReading());
    ClearICache();
  }
  else
  {
    sw.Do(&g_state.icache_tags);
    sw.Do(&g_state.icache_data);
  }

  if (sw.IsReading())
    UpdateFastmemBase();

  return !sw.HasError();
}

void UpdateFastmemBase()
{
  if (g_state.cop0_regs.sr.Isc)
    g_state.fastmem_base = nullptr;
  else
    g_state.fastmem_base = Bus::GetFastmemBase();
}

ALWAYS_INLINE_RELEASE void SetPC(u32 new_pc)
{
  DebugAssert(Common::IsAlignedPow2(new_pc, 4));
  g_state.regs.npc = new_pc;
  FlushPipeline(false);
}

ALWAYS_INLINE_RELEASE void Branch(u32 target)
{
  if (!Common::IsAlignedPow2(target, 4))
  {
    // The BadVaddr and EPC must be set to the fetching address, not the instruction about to execute.
    g_state.cop0_regs.BadVaddr = target;
    RaiseException(Cop0Registers::CAUSE::MakeValueForException(Exception::AdEL, false, false, 0), target);
    return;
  }

  g_state.regs.npc = target;
  g_state.branch_was_taken = true;
}

ALWAYS_INLINE static u32 GetExceptionVector(bool debug_exception = false)
{
  const u32 base = g_state.cop0_regs.sr.BEV ? UINT32_C(0xbfc00100) : UINT32_C(0x80000000);
  return base | (debug_exception ? UINT32_C(0x00000040) : UINT32_C(0x00000080));
}

ALWAYS_INLINE_RELEASE static void RaiseException(u32 CAUSE_bits, u32 EPC, u32 vector)
{
  g_state.cop0_regs.EPC = EPC;
  g_state.cop0_regs.cause.bits = (g_state.cop0_regs.cause.bits & ~Cop0Registers::CAUSE::EXCEPTION_WRITE_MASK) |
                                 (CAUSE_bits & Cop0Registers::CAUSE::EXCEPTION_WRITE_MASK);

  if (g_state.cop0_regs.cause.BD)
  {
    // TAR is set to the address which was being fetched in this instruction, or the next instruction to execute if the
    // exception hadn't occurred in the delay slot.
    g_state.cop0_regs.EPC -= UINT32_C(4);
    g_state.cop0_regs.TAR = g_state.regs.pc;
  }

  // current -> previous, switch to kernel mode and disable interrupts
  g_state.cop0_regs.sr.mode_bits <<= 2;

  // flush the pipeline - we don't want to execute the previously fetched instruction
  g_state.regs.npc = vector;
  g_state.exception_raised = true;
  FlushPipeline(true);
}

ALWAYS_INLINE_RELEASE static void DispatchCop0Breakpoint()
{
  // When a breakpoint address match occurs the PSX jumps to 80000040h (ie. unlike normal exceptions, not to 80000080h).
  // The Excode value in the CAUSE register is set to 09h (same as BREAK opcode), and EPC contains the return address,
  // as usually. One of the first things to be done in the exception handler is to disable breakpoints (eg. if the
  // any-jump break is enabled, then it must be disabled BEFORE jumping from 80000040h to the actual exception handler).
  RaiseException(Cop0Registers::CAUSE::MakeValueForException(
                   Exception::BP, g_state.current_instruction_in_branch_delay_slot,
                   g_state.current_instruction_was_branch_taken, g_state.current_instruction.cop.cop_n),
                 g_state.current_instruction_pc, GetExceptionVector(true));
}

void RaiseException(u32 CAUSE_bits, u32 EPC)
{
  RaiseException(CAUSE_bits, EPC, GetExceptionVector());
}

void RaiseException(Exception excode)
{
  RaiseException(Cop0Registers::CAUSE::MakeValueForException(excode, g_state.current_instruction_in_branch_delay_slot,
                                                             g_state.current_instruction_was_branch_taken,
                                                             g_state.current_instruction.cop.cop_n),
                 g_state.current_instruction_pc, GetExceptionVector());
}

void SetExternalInterrupt(u8 bit)
{
  g_state.cop0_regs.cause.Ip |= static_cast<u8>(1u << bit);

  if (g_settings.cpu_execution_mode == CPUExecutionMode::Interpreter)
  {
    g_state.interrupt_delay = 1;
  }
  else
  {
    g_state.interrupt_delay = 0;
    CheckForPendingInterrupt();
  }
}

void ClearExternalInterrupt(u8 bit)
{
  g_state.cop0_regs.cause.Ip &= static_cast<u8>(~(1u << bit));
}

ALWAYS_INLINE_RELEASE static void UpdateLoadDelay()
{
  // the old value is needed in case the delay slot instruction overwrites the same register
  if (g_state.load_delay_reg != Reg::count)
    g_state.regs.r[static_cast<u8>(g_state.load_delay_reg)] = g_state.load_delay_value;

  g_state.load_delay_reg = g_state.next_load_delay_reg;
  g_state.load_delay_value = g_state.next_load_delay_value;
  g_state.next_load_delay_reg = Reg::count;
}

ALWAYS_INLINE_RELEASE static void FlushPipeline(bool isException)
{
  // loads are flushed
  g_state.next_load_delay_reg = Reg::count;
  if (g_state.load_delay_reg != Reg::count)
  {
    g_state.regs.r[static_cast<u8>(g_state.load_delay_reg)] = g_state.load_delay_value;
    g_state.load_delay_reg = Reg::count;
  }

  // not in a branch delay slot
  g_state.branch_was_taken = false;
  g_state.next_instruction_is_branch_delay_slot = false;
  g_state.current_instruction_pc = g_state.regs.pc;

  if (isException)
  {
    tracer.isException = true;
    tracer.exceptionOpcode = g_state.next_instruction.bits;
  }

  // prefetch the next instruction
  FetchInstruction();
  if (isException)
  {
    TickCount oldticks = g_state.pending_ticks;
    g_state.regs.npc -= 4; FetchInstruction();
    g_state.regs.npc -= 4; FetchInstruction();
    if (oldticks == g_state.pending_ticks)
      g_state.pending_ticks += 2;
  }

  if (tracer.isInterrupt)
  {
      tracer.exceptionOpcode = g_state.next_instruction.bits;
  }

  // and set it as the next one to execute
  g_state.current_instruction.bits = g_state.next_instruction.bits;
  g_state.current_instruction_in_branch_delay_slot = false;
  g_state.current_instruction_was_branch_taken = false;
}

ALWAYS_INLINE static u32 ReadReg(Reg rs)
{
  return g_state.regs.r[static_cast<u8>(rs)];
}

ALWAYS_INLINE static void WriteReg(Reg rd, u32 value)
{
  g_state.regs.r[static_cast<u8>(rd)] = value;
  g_state.load_delay_reg = (rd == g_state.load_delay_reg) ? Reg::count : g_state.load_delay_reg;

  // prevent writes to $zero from going through - better than branching/cmov
  g_state.regs.zero = 0;
}

ALWAYS_INLINE_RELEASE static void WriteRegDelayed(Reg rd, u32 value)
{
  Assert(g_state.next_load_delay_reg == Reg::count);
  if (rd == Reg::zero)
    return;

  // double load delays ignore the first value
  if (g_state.load_delay_reg == rd)
    g_state.load_delay_reg = Reg::count;

  // save the old value, if something else overwrites this reg we want to preserve it
  g_state.next_load_delay_reg = rd;
  g_state.next_load_delay_value = value;
}

ALWAYS_INLINE_RELEASE static u32 ReadCop0Reg(Cop0Reg reg)
{
  switch (reg)
  {
    case Cop0Reg::BPC:
      return g_state.cop0_regs.BPC;

    case Cop0Reg::BPCM:
      return g_state.cop0_regs.BPCM;

    case Cop0Reg::BDA:
      return g_state.cop0_regs.BDA;

    case Cop0Reg::BDAM:
      return g_state.cop0_regs.BDAM;

    case Cop0Reg::DCIC:
      return g_state.cop0_regs.dcic.bits;

    case Cop0Reg::JUMPDEST:
      return g_state.cop0_regs.TAR;

    case Cop0Reg::BadVaddr:
      return g_state.cop0_regs.BadVaddr;

    case Cop0Reg::SR:
      return g_state.cop0_regs.sr.bits;

    case Cop0Reg::CAUSE:
      return g_state.cop0_regs.cause.bits;

    case Cop0Reg::EPC:
      return g_state.cop0_regs.EPC;

    case Cop0Reg::PRID:
      return g_state.cop0_regs.PRID;

    default:
      return 0;
  }
}

ALWAYS_INLINE_RELEASE static void WriteCop0Reg(Cop0Reg reg, u32 value)
{
  switch (reg)
  {
    case Cop0Reg::BPC:
    {
      g_state.cop0_regs.BPC = value;
      Log_DevPrintf("COP0 BPC <- %08X", value);
    }
    break;

    case Cop0Reg::BPCM:
    {
      g_state.cop0_regs.BPCM = value;
      Log_DevPrintf("COP0 BPCM <- %08X", value);
    }
    break;

    case Cop0Reg::BDA:
    {
      g_state.cop0_regs.BDA = value;
      Log_DevPrintf("COP0 BDA <- %08X", value);
    }
    break;

    case Cop0Reg::BDAM:
    {
      g_state.cop0_regs.BDAM = value;
      Log_DevPrintf("COP0 BDAM <- %08X", value);
    }
    break;

    case Cop0Reg::JUMPDEST:
    {
      Log_WarningPrintf("Ignoring write to Cop0 JUMPDEST");
    }
    break;

    case Cop0Reg::DCIC:
    {
      g_state.cop0_regs.dcic.bits =
        (g_state.cop0_regs.dcic.bits & ~Cop0Registers::DCIC::WRITE_MASK) | (value & Cop0Registers::DCIC::WRITE_MASK);
      Log_DevPrintf("COP0 DCIC <- %08X (now %08X)", value, g_state.cop0_regs.dcic.bits);
      UpdateDebugDispatcherFlag();
    }
    break;

    case Cop0Reg::SR:
    {
      g_state.cop0_regs.sr.bits =
        (g_state.cop0_regs.sr.bits & ~Cop0Registers::SR::WRITE_MASK) | (value & Cop0Registers::SR::WRITE_MASK);
      Log_DebugPrintf("COP0 SR <- %08X (now %08X)", value, g_state.cop0_regs.sr.bits);
      CheckForPendingInterrupt();
    }
    break;

    case Cop0Reg::CAUSE:
    {
      g_state.cop0_regs.cause.bits =
        (g_state.cop0_regs.cause.bits & ~Cop0Registers::CAUSE::WRITE_MASK) | (value & Cop0Registers::CAUSE::WRITE_MASK);
      Log_DebugPrintf("COP0 CAUSE <- %08X (now %08X)", value, g_state.cop0_regs.cause.bits);
      CheckForPendingInterrupt();
    }
    break;

    default:
      Log_DevPrintf("Unknown COP0 reg write %u (%08X)", ZeroExtend32(static_cast<u8>(reg)), value);
      break;
  }
}

ALWAYS_INLINE_RELEASE void Cop0ExecutionBreakpointCheck()
{
  if (!g_state.cop0_regs.dcic.ExecutionBreakpointsEnabled())
    return;

  const u32 pc = g_state.current_instruction_pc;
  const u32 bpc = g_state.cop0_regs.BPC;
  const u32 bpcm = g_state.cop0_regs.BPCM;

  // Break condition is "((PC XOR BPC) AND BPCM)=0".
  if (bpcm == 0 || ((pc ^ bpc) & bpcm) != 0u)
    return;

  Log_DevPrintf("Cop0 execution breakpoint at %08X", pc);
  g_state.cop0_regs.dcic.status_any_break = true;
  g_state.cop0_regs.dcic.status_bpc_code_break = true;
  DispatchCop0Breakpoint();
}

template<MemoryAccessType type>
ALWAYS_INLINE_RELEASE void Cop0DataBreakpointCheck(VirtualMemoryAddress address)
{
  if constexpr (type == MemoryAccessType::Read)
  {
    if (!g_state.cop0_regs.dcic.DataReadBreakpointsEnabled())
      return;
  }
  else
  {
    if (!g_state.cop0_regs.dcic.DataWriteBreakpointsEnabled())
      return;
  }

  // Break condition is "((addr XOR BDA) AND BDAM)=0".
  const u32 bda = g_state.cop0_regs.BDA;
  const u32 bdam = g_state.cop0_regs.BDAM;
  if (bdam == 0 || ((address ^ bda) & bdam) != 0u)
    return;

  Log_DevPrintf("Cop0 data breakpoint for %08X at %08X", address, g_state.current_instruction_pc);

  g_state.cop0_regs.dcic.status_any_break = true;
  g_state.cop0_regs.dcic.status_bda_data_break = true;
  if constexpr (type == MemoryAccessType::Read)
    g_state.cop0_regs.dcic.status_bda_data_read_break = true;
  else
    g_state.cop0_regs.dcic.status_bda_data_write_break = true;

  DispatchCop0Breakpoint();
}

static void TracePrintInstruction()
{
  const u32 pc = g_state.current_instruction_pc;
  const u32 bits = g_state.current_instruction.bits;

  TinyString instr;
  TinyString comment;
  DisassembleInstruction(&instr, pc, bits);
  DisassembleInstructionComment(&comment, pc, bits, &g_state.regs);
  if (!comment.IsEmpty())
  {
    for (u32 i = instr.GetLength(); i < 30; i++)
      instr.AppendCharacter(' ');
    instr.AppendString("; ");
    instr.AppendString(comment);
  }

  std::printf("%08x: %08x %s\n", pc, bits, instr.GetCharArray());
}

static void PrintInstruction(u32 bits, u32 pc, Registers* regs, const char* prefix)
{
  TinyString instr;
  TinyString comment;
  DisassembleInstruction(&instr, pc, bits);
  DisassembleInstructionComment(&comment, pc, bits, regs);
  if (!comment.IsEmpty())
  {
    for (u32 i = instr.GetLength(); i < 30; i++)
      instr.AppendCharacter(' ');
    instr.AppendString("; ");
    instr.AppendString(comment);
  }

  Log_DevPrintf("%s%08x: %08x %s", prefix, pc, bits, instr.GetCharArray());
}

static void LogInstruction(u32 bits, u32 pc, Registers* regs)
{
  TinyString instr;
  TinyString comment;
  DisassembleInstruction(&instr, pc, bits);
  DisassembleInstructionComment(&comment, pc, bits, regs);
  if (!comment.IsEmpty())
  {
    for (u32 i = instr.GetLength(); i < 30; i++)
      instr.AppendCharacter(' ');
    instr.AppendString("; ");
    instr.AppendString(comment);
  }

  WriteToExecutionLog("%08x: %08x %s\n", pc, bits, instr.GetCharArray());
}

ALWAYS_INLINE static constexpr bool AddOverflow(u32 old_value, u32 add_value, u32 new_value)
{
  return (((new_value ^ old_value) & (new_value ^ add_value)) & UINT32_C(0x80000000)) != 0;
}

ALWAYS_INLINE static constexpr bool SubOverflow(u32 old_value, u32 sub_value, u32 new_value)
{
  return (((new_value ^ old_value) & (old_value ^ sub_value)) & UINT32_C(0x80000000)) != 0;
}

void DisassembleAndPrint(u32 addr, const char* prefix)
{
  u32 bits = 0;
  SafeReadMemoryWord(addr, &bits);
  PrintInstruction(bits, addr, &g_state.regs, prefix);
}

void DisassembleAndPrint(u32 addr, u32 instructions_before /* = 0 */, u32 instructions_after /* = 0 */)
{
  u32 disasm_addr = addr - (instructions_before * sizeof(u32));
  for (u32 i = 0; i < instructions_before; i++)
  {
    DisassembleAndPrint(disasm_addr, "");
    disasm_addr += sizeof(u32);
  }

  // <= to include the instruction itself
  for (u32 i = 0; i <= instructions_after; i++)
  {
    DisassembleAndPrint(disasm_addr, (i == 0) ? "---->" : "");
    disasm_addr += sizeof(u32);
  }
}

template<PGXPMode pgxp_mode, bool debug>
ALWAYS_INLINE_RELEASE static void ExecuteInstruction()
{
restart_instruction:
  const Instruction inst = g_state.current_instruction;

  // Skip nops. Makes PGXP-CPU quicker, but also the regular interpreter.
  if (inst.bits == 0)
    return;

  switch (inst.op)
  {
    case InstructionOp::funct:
    {
      switch (inst.r.funct)
      {
        case InstructionFunct::sll:
        {
          const u32 new_value = ReadReg(inst.r.rt) << inst.r.shamt;
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_SLL(inst.bits, ReadReg(inst.r.rt));

          WriteReg(inst.r.rd, new_value);
        }
        break;

        case InstructionFunct::srl:
        {
          const u32 new_value = ReadReg(inst.r.rt) >> inst.r.shamt;
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_SRL(inst.bits, ReadReg(inst.r.rt));

          WriteReg(inst.r.rd, new_value);
        }
        break;

        case InstructionFunct::sra:
        {
          const u32 new_value = static_cast<u32>(static_cast<s32>(ReadReg(inst.r.rt)) >> inst.r.shamt);
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_SRA(inst.bits, ReadReg(inst.r.rt));

          WriteReg(inst.r.rd, new_value);
        }
        break;

        case InstructionFunct::sllv:
        {
          const u32 shift_amount = ReadReg(inst.r.rs) & UINT32_C(0x1F);
          const u32 new_value = ReadReg(inst.r.rt) << shift_amount;
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_SLLV(inst.bits, ReadReg(inst.r.rt), shift_amount);

          WriteReg(inst.r.rd, new_value);
        }
        break;

        case InstructionFunct::srlv:
        {
          const u32 shift_amount = ReadReg(inst.r.rs) & UINT32_C(0x1F);
          const u32 new_value = ReadReg(inst.r.rt) >> shift_amount;
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_SRLV(inst.bits, ReadReg(inst.r.rt), shift_amount);

          WriteReg(inst.r.rd, new_value);
        }
        break;

        case InstructionFunct::srav:
        {
          const u32 shift_amount = ReadReg(inst.r.rs) & UINT32_C(0x1F);
          const u32 new_value = static_cast<u32>(static_cast<s32>(ReadReg(inst.r.rt)) >> shift_amount);
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_SRAV(inst.bits, ReadReg(inst.r.rt), shift_amount);

          WriteReg(inst.r.rd, new_value);
        }
        break;

        case InstructionFunct::and_:
        {
          const u32 new_value = ReadReg(inst.r.rs) & ReadReg(inst.r.rt);
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_AND_(inst.bits, ReadReg(inst.r.rs), ReadReg(inst.r.rt));

          WriteReg(inst.r.rd, new_value);
        }
        break;

        case InstructionFunct::or_:
        {
          const u32 new_value = ReadReg(inst.r.rs) | ReadReg(inst.r.rt);
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_OR_(inst.bits, ReadReg(inst.r.rs), ReadReg(inst.r.rt));

          WriteReg(inst.r.rd, new_value);
        }
        break;

        case InstructionFunct::xor_:
        {
          const u32 new_value = ReadReg(inst.r.rs) ^ ReadReg(inst.r.rt);
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_XOR_(inst.bits, ReadReg(inst.r.rs), ReadReg(inst.r.rt));

          WriteReg(inst.r.rd, new_value);
        }
        break;

        case InstructionFunct::nor:
        {
          const u32 new_value = ~(ReadReg(inst.r.rs) | ReadReg(inst.r.rt));
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_NOR(inst.bits, ReadReg(inst.r.rs), ReadReg(inst.r.rt));

          WriteReg(inst.r.rd, new_value);
        }
        break;

        case InstructionFunct::add:
        {
          const u32 old_value = ReadReg(inst.r.rs);
          const u32 add_value = ReadReg(inst.r.rt);
          const u32 new_value = old_value + add_value;
          if (AddOverflow(old_value, add_value, new_value))
          {
            RaiseException(Exception::Ov);
            return;
          }

          if constexpr (pgxp_mode == PGXPMode::CPU)
            PGXP::CPU_ADD(inst.bits, ReadReg(inst.r.rs), ReadReg(inst.r.rt));
          else if constexpr (pgxp_mode >= PGXPMode::Memory)
          {
            if (add_value == 0)
            {
              PGXP::CPU_MOVE((static_cast<u32>(inst.r.rd.GetValue()) << 8) | static_cast<u32>(inst.r.rs.GetValue()),
                             old_value);
            }
          }

          WriteReg(inst.r.rd, new_value);
        }
        break;

        case InstructionFunct::addu:
        {
          const u32 old_value = ReadReg(inst.r.rs);
          const u32 add_value = ReadReg(inst.r.rt);
          const u32 new_value = old_value + add_value;
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_ADD(inst.bits, old_value, add_value);
          else if constexpr (pgxp_mode >= PGXPMode::Memory)
          {
            if (add_value == 0)
            {
              PGXP::CPU_MOVE((static_cast<u32>(inst.r.rd.GetValue()) << 8) | static_cast<u32>(inst.r.rs.GetValue()),
                             old_value);
            }
          }

          WriteReg(inst.r.rd, new_value);
        }
        break;

        case InstructionFunct::sub:
        {
          const u32 old_value = ReadReg(inst.r.rs);
          const u32 sub_value = ReadReg(inst.r.rt);
          const u32 new_value = old_value - sub_value;
          if (SubOverflow(old_value, sub_value, new_value))
          {
            RaiseException(Exception::Ov);
            return;
          }

          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_SUB(inst.bits, ReadReg(inst.r.rs), ReadReg(inst.r.rt));

          WriteReg(inst.r.rd, new_value);
        }
        break;

        case InstructionFunct::subu:
        {
          const u32 new_value = ReadReg(inst.r.rs) - ReadReg(inst.r.rt);
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_SUB(inst.bits, ReadReg(inst.r.rs), ReadReg(inst.r.rt));

          WriteReg(inst.r.rd, new_value);
        }
        break;

        case InstructionFunct::slt:
        {
          const u32 result = BoolToUInt32(static_cast<s32>(ReadReg(inst.r.rs)) < static_cast<s32>(ReadReg(inst.r.rt)));
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_SLT(inst.bits, ReadReg(inst.r.rs), ReadReg(inst.r.rt));

          WriteReg(inst.r.rd, result);
        }
        break;

        case InstructionFunct::sltu:
        {
          const u32 result = BoolToUInt32(ReadReg(inst.r.rs) < ReadReg(inst.r.rt));
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_SLTU(inst.bits, ReadReg(inst.r.rs), ReadReg(inst.r.rt));

          WriteReg(inst.r.rd, result);
        }
        break;

        case InstructionFunct::mfhi:
        {
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_MFHI(inst.bits, g_state.regs.hi);

          WriteReg(inst.r.rd, g_state.regs.hi);
        }
        break;

        case InstructionFunct::mthi:
        {
          const u32 value = ReadReg(inst.r.rs);
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_MTHI(inst.bits, value);

          g_state.regs.hi = value;
        }
        break;

        case InstructionFunct::mflo:
        {
          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_MFLO(inst.bits, g_state.regs.lo);

          WriteReg(inst.r.rd, g_state.regs.lo);
        }
        break;

        case InstructionFunct::mtlo:
        {
          const u32 value = ReadReg(inst.r.rs);
          if constexpr (pgxp_mode == PGXPMode::CPU)
            PGXP::CPU_MTLO(inst.bits, value);

          g_state.regs.lo = value;
        }
        break;

        case InstructionFunct::mult:
        {
          const u32 lhs = ReadReg(inst.r.rs);
          const u32 rhs = ReadReg(inst.r.rt);
          const u64 result =
            static_cast<u64>(static_cast<s64>(SignExtend64(lhs)) * static_cast<s64>(SignExtend64(rhs)));

          g_state.regs.hi = Truncate32(result >> 32);
          g_state.regs.lo = Truncate32(result);

          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_MULT(inst.bits, lhs, rhs);
        }
        break;

        case InstructionFunct::multu:
        {
          const u32 lhs = ReadReg(inst.r.rs);
          const u32 rhs = ReadReg(inst.r.rt);
          const u64 result = ZeroExtend64(lhs) * ZeroExtend64(rhs);

          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_MULTU(inst.bits, lhs, rhs);

          g_state.regs.hi = Truncate32(result >> 32);
          g_state.regs.lo = Truncate32(result);
        }
        break;

        case InstructionFunct::div:
        {
          const s32 num = static_cast<s32>(ReadReg(inst.r.rs));
          const s32 denom = static_cast<s32>(ReadReg(inst.r.rt));

          if (denom == 0)
          {
            // divide by zero
            g_state.regs.lo = (num >= 0) ? UINT32_C(0xFFFFFFFF) : UINT32_C(1);
            g_state.regs.hi = static_cast<u32>(num);
          }
          else if (static_cast<u32>(num) == UINT32_C(0x80000000) && denom == -1)
          {
            // unrepresentable
            g_state.regs.lo = UINT32_C(0x80000000);
            g_state.regs.hi = 0;
          }
          else
          {
            g_state.regs.lo = static_cast<u32>(num / denom);
            g_state.regs.hi = static_cast<u32>(num % denom);
          }

          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_DIV(inst.bits, num, denom);
        }
        break;

        case InstructionFunct::divu:
        {
          const u32 num = ReadReg(inst.r.rs);
          const u32 denom = ReadReg(inst.r.rt);

          if (denom == 0)
          {
            // divide by zero
            g_state.regs.lo = UINT32_C(0xFFFFFFFF);
            g_state.regs.hi = static_cast<u32>(num);
          }
          else
          {
            g_state.regs.lo = num / denom;
            g_state.regs.hi = num % denom;
          }

          if constexpr (pgxp_mode >= PGXPMode::CPU)
            PGXP::CPU_DIVU(inst.bits, num, denom);
        }
        break;

        case InstructionFunct::jr:
        {
          g_state.next_instruction_is_branch_delay_slot = true;
          const u32 target = ReadReg(inst.r.rs);
          Branch(target);
        }
        break;

        case InstructionFunct::jalr:
        {
          g_state.next_instruction_is_branch_delay_slot = true;
          const u32 target = ReadReg(inst.r.rs);
          WriteReg(inst.r.rd, g_state.regs.npc);
          Branch(target);
        }
        break;

        case InstructionFunct::syscall:
        {
          RaiseException(Exception::Syscall);
        }
        break;

        case InstructionFunct::break_:
        {
          RaiseException(Exception::BP);
        }
        break;

        default:
        {
          RaiseException(Exception::RI);
          break;
        }
      }
    }
    break;

    case InstructionOp::lui:
    {
      const u32 value = inst.i.imm_zext32() << 16;
      WriteReg(inst.i.rt, value);

      if constexpr (pgxp_mode >= PGXPMode::CPU)
        PGXP::CPU_LUI(inst.bits);
    }
    break;

    case InstructionOp::andi:
    {
      const u32 new_value = ReadReg(inst.i.rs) & inst.i.imm_zext32();

      if constexpr (pgxp_mode >= PGXPMode::CPU)
        PGXP::CPU_ANDI(inst.bits, ReadReg(inst.i.rs));

      WriteReg(inst.i.rt, new_value);
    }
    break;

    case InstructionOp::ori:
    {
      const u32 new_value = ReadReg(inst.i.rs) | inst.i.imm_zext32();

      if constexpr (pgxp_mode >= PGXPMode::CPU)
        PGXP::CPU_ORI(inst.bits, ReadReg(inst.i.rs));

      WriteReg(inst.i.rt, new_value);
    }
    break;

    case InstructionOp::xori:
    {
      const u32 new_value = ReadReg(inst.i.rs) ^ inst.i.imm_zext32();

      if constexpr (pgxp_mode >= PGXPMode::CPU)
        PGXP::CPU_XORI(inst.bits, ReadReg(inst.i.rs));

      WriteReg(inst.i.rt, new_value);
    }
    break;

    case InstructionOp::addi:
    {
      const u32 old_value = ReadReg(inst.i.rs);
      const u32 add_value = inst.i.imm_sext32();
      const u32 new_value = old_value + add_value;
      if (AddOverflow(old_value, add_value, new_value))
      {
        RaiseException(Exception::Ov);
        return;
      }

      if constexpr (pgxp_mode >= PGXPMode::CPU)
        PGXP::CPU_ADDI(inst.bits, ReadReg(inst.i.rs));
      else if constexpr (pgxp_mode >= PGXPMode::Memory)
      {
        if (add_value == 0)
        {
          PGXP::CPU_MOVE((static_cast<u32>(inst.i.rt.GetValue()) << 8) | static_cast<u32>(inst.i.rs.GetValue()),
                         old_value);
        }
      }

      WriteReg(inst.i.rt, new_value);
    }
    break;

    case InstructionOp::addiu:
    {
      const u32 old_value = ReadReg(inst.i.rs);
      const u32 add_value = inst.i.imm_sext32();
      const u32 new_value = old_value + add_value;

      if constexpr (pgxp_mode >= PGXPMode::CPU)
        PGXP::CPU_ADDI(inst.bits, ReadReg(inst.i.rs));
      else if constexpr (pgxp_mode >= PGXPMode::Memory)
      {
        if (add_value == 0)
        {
          PGXP::CPU_MOVE((static_cast<u32>(inst.i.rt.GetValue()) << 8) | static_cast<u32>(inst.i.rs.GetValue()),
                         old_value);
        }
      }

      WriteReg(inst.i.rt, new_value);
    }
    break;

    case InstructionOp::slti:
    {
      const u32 result = BoolToUInt32(static_cast<s32>(ReadReg(inst.i.rs)) < static_cast<s32>(inst.i.imm_sext32()));

      if constexpr (pgxp_mode >= PGXPMode::CPU)
        PGXP::CPU_SLTI(inst.bits, ReadReg(inst.i.rs));

      WriteReg(inst.i.rt, result);
    }
    break;

    case InstructionOp::sltiu:
    {
      const u32 result = BoolToUInt32(ReadReg(inst.i.rs) < inst.i.imm_sext32());

      if constexpr (pgxp_mode >= PGXPMode::CPU)
        PGXP::CPU_SLTIU(inst.bits, ReadReg(inst.i.rs));

      WriteReg(inst.i.rt, result);
    }
    break;

    case InstructionOp::lb:
    {
      const VirtualMemoryAddress addr = ReadReg(inst.i.rs) + inst.i.imm_sext32();
      if constexpr (debug)
        Cop0DataBreakpointCheck<MemoryAccessType::Read>(addr);

      u8 value;
      if (!ReadMemoryByte(addr, &value))
        return;

      const u32 sxvalue = SignExtend32(value);

      WriteRegDelayed(inst.i.rt, sxvalue);

      if constexpr (pgxp_mode >= PGXPMode::Memory)
        PGXP::CPU_LBx(inst.bits, sxvalue, addr);
    }
    break;

    case InstructionOp::lh:
    {
      const VirtualMemoryAddress addr = ReadReg(inst.i.rs) + inst.i.imm_sext32();
      if constexpr (debug)
        Cop0DataBreakpointCheck<MemoryAccessType::Read>(addr);

      u16 value;
      if (!ReadMemoryHalfWord(addr, &value))
        return;

      const u32 sxvalue = SignExtend32(value);
      WriteRegDelayed(inst.i.rt, sxvalue);

      if constexpr (pgxp_mode >= PGXPMode::Memory)
        PGXP::CPU_LHx(inst.bits, sxvalue, addr);
    }
    break;

    case InstructionOp::lw:
    {
      const VirtualMemoryAddress addr = ReadReg(inst.i.rs) + inst.i.imm_sext32();
      if constexpr (debug)
        Cop0DataBreakpointCheck<MemoryAccessType::Read>(addr);

      u32 value;
      if (!ReadMemoryWord(addr, &value))
        return;

      WriteRegDelayed(inst.i.rt, value);

      if constexpr (pgxp_mode >= PGXPMode::Memory)
        PGXP::CPU_LW(inst.bits, value, addr);
    }
    break;

    case InstructionOp::lbu:
    {
      const VirtualMemoryAddress addr = ReadReg(inst.i.rs) + inst.i.imm_sext32();
      if constexpr (debug)
        Cop0DataBreakpointCheck<MemoryAccessType::Read>(addr);

      u8 value;
      if (!ReadMemoryByte(addr, &value))
        return;

      const u32 zxvalue = ZeroExtend32(value);
      WriteRegDelayed(inst.i.rt, zxvalue);

      if constexpr (pgxp_mode >= PGXPMode::Memory)
        PGXP::CPU_LBx(inst.bits, zxvalue, addr);
    }
    break;

    case InstructionOp::lhu:
    {
      const VirtualMemoryAddress addr = ReadReg(inst.i.rs) + inst.i.imm_sext32();
      if constexpr (debug)
        Cop0DataBreakpointCheck<MemoryAccessType::Read>(addr);

      u16 value;
      if (!ReadMemoryHalfWord(addr, &value))
        return;

      const u32 zxvalue = ZeroExtend32(value);
      WriteRegDelayed(inst.i.rt, zxvalue);

      if constexpr (pgxp_mode >= PGXPMode::Memory)
        PGXP::CPU_LHx(inst.bits, zxvalue, addr);
    }
    break;

    case InstructionOp::lwl:
    case InstructionOp::lwr:
    {
      const VirtualMemoryAddress addr = ReadReg(inst.i.rs) + inst.i.imm_sext32();
      const VirtualMemoryAddress aligned_addr = addr & ~UINT32_C(3);
      if constexpr (debug)
        Cop0DataBreakpointCheck<MemoryAccessType::Read>(addr);

      u32 aligned_value;
      if (!ReadMemoryWord(aligned_addr, &aligned_value))
        return;

      // Bypasses load delay. No need to check the old value since this is the delay slot or it's not relevant.
      const u32 existing_value = (inst.i.rt == g_state.load_delay_reg) ? g_state.load_delay_value : ReadReg(inst.i.rt);
      const u8 shift = (Truncate8(addr) & u8(3)) * u8(8);
      u32 new_value;
      if (inst.op == InstructionOp::lwl)
      {
        const u32 mask = UINT32_C(0x00FFFFFF) >> shift;
        new_value = (existing_value & mask) | (aligned_value << (24 - shift));
      }
      else
      {
        const u32 mask = UINT32_C(0xFFFFFF00) << (24 - shift);
        new_value = (existing_value & mask) | (aligned_value >> shift);
      }

      WriteRegDelayed(inst.i.rt, new_value);

      if constexpr (pgxp_mode >= PGXPMode::Memory)
        PGXP::CPU_LW(inst.bits, new_value, addr);
    }
    break;

    case InstructionOp::sb:
    {
      const VirtualMemoryAddress addr = ReadReg(inst.i.rs) + inst.i.imm_sext32();
      if constexpr (debug)
        Cop0DataBreakpointCheck<MemoryAccessType::Write>(addr);

      const u32 value = ReadReg(inst.i.rt);
      WriteMemoryByte(addr, value);

      if constexpr (pgxp_mode >= PGXPMode::Memory)
        PGXP::CPU_SB(inst.bits, Truncate8(value), addr);
    }
    break;

    case InstructionOp::sh:
    {
      const VirtualMemoryAddress addr = ReadReg(inst.i.rs) + inst.i.imm_sext32();
      if constexpr (debug)
        Cop0DataBreakpointCheck<MemoryAccessType::Write>(addr);

      const u32 value = ReadReg(inst.i.rt);
      WriteMemoryHalfWord(addr, value);

      if constexpr (pgxp_mode >= PGXPMode::Memory)
        PGXP::CPU_SH(inst.bits, Truncate16(value), addr);
    }
    break;

    case InstructionOp::sw:
    {
      const VirtualMemoryAddress addr = ReadReg(inst.i.rs) + inst.i.imm_sext32();
      if constexpr (debug)
        Cop0DataBreakpointCheck<MemoryAccessType::Write>(addr);

      const u32 value = ReadReg(inst.i.rt);
      WriteMemoryWord(addr, value);

      if constexpr (pgxp_mode >= PGXPMode::Memory)
        PGXP::CPU_SW(inst.bits, value, addr);
    }
    break;

    case InstructionOp::swl:
    case InstructionOp::swr:
    {
      const VirtualMemoryAddress addr = ReadReg(inst.i.rs) + inst.i.imm_sext32();
      const VirtualMemoryAddress aligned_addr = addr & ~UINT32_C(3);
      if constexpr (debug)
        Cop0DataBreakpointCheck<MemoryAccessType::Write>(aligned_addr);

      const u32 reg_value = ReadReg(inst.i.rt);
      const u8 shift = (Truncate8(addr) & u8(3)) * u8(8);
      u32 mem_value;
      if (!ReadMemoryWord(aligned_addr, &mem_value))
        return;

      u32 new_value;
      if (inst.op == InstructionOp::swl)
      {
        const u32 mem_mask = UINT32_C(0xFFFFFF00) << shift;
        new_value = (mem_value & mem_mask) | (reg_value >> (24 - shift));
      }
      else
      {
        const u32 mem_mask = UINT32_C(0x00FFFFFF) >> (24 - shift);
        new_value = (mem_value & mem_mask) | (reg_value << shift);
      }

      WriteMemoryWord(aligned_addr, new_value);

      if constexpr (pgxp_mode >= PGXPMode::Memory)
        PGXP::CPU_SW(inst.bits, new_value, addr);
    }
    break;

    case InstructionOp::j:
    {
      g_state.next_instruction_is_branch_delay_slot = true;
      Branch((g_state.regs.pc & UINT32_C(0xF0000000)) | (inst.j.target << 2));
    }
    break;

    case InstructionOp::jal:
    {
      WriteReg(Reg::ra, g_state.regs.npc);
      g_state.next_instruction_is_branch_delay_slot = true;
      Branch((g_state.regs.pc & UINT32_C(0xF0000000)) | (inst.j.target << 2));
    }
    break;

    case InstructionOp::beq:
    {
      // We're still flagged as a branch delay slot even if the branch isn't taken.
      g_state.next_instruction_is_branch_delay_slot = true;
      const bool branch = (ReadReg(inst.i.rs) == ReadReg(inst.i.rt));
      if (branch)
        Branch(g_state.regs.pc + (inst.i.imm_sext32() << 2));
    }
    break;

    case InstructionOp::bne:
    {
      g_state.next_instruction_is_branch_delay_slot = true;
      const bool branch = (ReadReg(inst.i.rs) != ReadReg(inst.i.rt));
      u32 addr = g_state.regs.pc + (inst.i.imm_sext32() << 2);
      if (addr == 0xbfc06be8)
      {
          int a = 5;
      }
      if (branch)
        Branch(g_state.regs.pc + (inst.i.imm_sext32() << 2));
    }
    break;

    case InstructionOp::bgtz:
    {
      g_state.next_instruction_is_branch_delay_slot = true;
      const bool branch = (static_cast<s32>(ReadReg(inst.i.rs)) > 0);
      if (branch)
        Branch(g_state.regs.pc + (inst.i.imm_sext32() << 2));
    }
    break;

    case InstructionOp::blez:
    {
      g_state.next_instruction_is_branch_delay_slot = true;
      const bool branch = (static_cast<s32>(ReadReg(inst.i.rs)) <= 0);
      if (branch)
        Branch(g_state.regs.pc + (inst.i.imm_sext32() << 2));
    }
    break;

    case InstructionOp::b:
    {
      g_state.next_instruction_is_branch_delay_slot = true;
      const u8 rt = static_cast<u8>(inst.i.rt.GetValue());

      // bgez is the inverse of bltz, so simply do ltz and xor the result
      const bool bgez = ConvertToBoolUnchecked(rt & u8(1));
      const bool branch = (static_cast<s32>(ReadReg(inst.i.rs)) < 0) ^ bgez;

      // register is still linked even if the branch isn't taken
      const bool link = (rt & u8(0x1E)) == u8(0x10);
      if (link)
        WriteReg(Reg::ra, g_state.regs.npc);

      if (branch)
        Branch(g_state.regs.pc + (inst.i.imm_sext32() << 2));
    }
    break;

    case InstructionOp::cop0:
    {
      if (InUserMode() && !g_state.cop0_regs.sr.CU0)
      {
        Log_WarningPrintf("Coprocessor 0 not present in user mode");
        RaiseException(Exception::CpU);
        return;
      }

      if (inst.cop.IsCommonInstruction())
      {
        switch (inst.cop.CommonOp())
        {
          case CopCommonInstruction::mfcn:
          {
            const u32 value = ReadCop0Reg(static_cast<Cop0Reg>(inst.r.rd.GetValue()));

            if constexpr (pgxp_mode == PGXPMode::CPU)
              PGXP::CPU_MFC0(inst.bits, value);

            WriteRegDelayed(inst.r.rt, value);
          }
          break;

          case CopCommonInstruction::mtcn:
          {
            WriteCop0Reg(static_cast<Cop0Reg>(inst.r.rd.GetValue()), ReadReg(inst.r.rt));

            if constexpr (pgxp_mode == PGXPMode::CPU)
              PGXP::CPU_MTC0(inst.bits, ReadCop0Reg(static_cast<Cop0Reg>(inst.r.rd.GetValue())), ReadReg(inst.i.rt));
          }
          break;

          default:
            Log_ErrorPrintf("Unhandled instruction at %08X: %08X", g_state.current_instruction_pc, inst.bits);
            break;
        }
      }
      else
      {
        switch (inst.cop.Cop0Op())
        {
          case Cop0Instruction::rfe:
          {
            // restore mode
            g_state.cop0_regs.sr.mode_bits =
              (g_state.cop0_regs.sr.mode_bits & UINT32_C(0b110000)) | (g_state.cop0_regs.sr.mode_bits >> 2);
            CheckForPendingInterrupt();
          }
          break;

          case Cop0Instruction::tlbr:
          case Cop0Instruction::tlbwi:
          case Cop0Instruction::tlbwr:
          case Cop0Instruction::tlbp:
            RaiseException(Exception::RI);
            break;

          default:
            Log_ErrorPrintf("Unhandled instruction at %08X: %08X", g_state.current_instruction_pc, inst.bits);
            break;
        }
      }
    }
    break;

    case InstructionOp::cop2:
    {
      if (!g_state.cop0_regs.sr.CE2)
      {
        Log_WarningPrintf("Coprocessor 2 not enabled");
        RaiseException(Exception::CpU);
        return;
      }

      if (inst.cop.IsCommonInstruction())
      {
        // TODO: Combine with cop0.
        switch (inst.cop.CommonOp())
        {
          case CopCommonInstruction::cfcn:
          {
            const u32 value = GTE::ReadRegister(static_cast<u32>(inst.r.rd.GetValue()) + 32);
            WriteRegDelayed(inst.r.rt, value);

            if constexpr (pgxp_mode >= PGXPMode::Memory)
              PGXP::CPU_CFC2(inst.bits, value, value);
          }
          break;

          case CopCommonInstruction::ctcn:
          {
            const u32 value = ReadReg(inst.r.rt);
            GTE::WriteRegister(static_cast<u32>(inst.r.rd.GetValue()) + 32, value);

            if constexpr (pgxp_mode >= PGXPMode::Memory)
              PGXP::CPU_CTC2(inst.bits, value, value);
          }
          break;

          case CopCommonInstruction::mfcn:
          {
            const u32 value = GTE::ReadRegister(static_cast<u32>(inst.r.rd.GetValue()));
            WriteRegDelayed(inst.r.rt, value);

            if constexpr (pgxp_mode >= PGXPMode::Memory)
              PGXP::CPU_MFC2(inst.bits, value, value);
          }
          break;

          case CopCommonInstruction::mtcn:
          {
            const u32 value = ReadReg(inst.r.rt);
            GTE::WriteRegister(static_cast<u32>(inst.r.rd.GetValue()), value);

            if constexpr (pgxp_mode >= PGXPMode::Memory)
              PGXP::CPU_MTC2(inst.bits, value, value);
          }
          break;

          default:
            Log_ErrorPrintf("Unhandled instruction at %08X: %08X", g_state.current_instruction_pc, inst.bits);
            break;
        }
      }
      else
      {
        GTE::ExecuteInstruction(inst.bits);
      }
    }
    break;

    case InstructionOp::lwc2:
    {
      if (!g_state.cop0_regs.sr.CE2)
      {
        Log_WarningPrintf("Coprocessor 2 not enabled");
        RaiseException(Exception::CpU);
        return;
      }

      const VirtualMemoryAddress addr = ReadReg(inst.i.rs) + inst.i.imm_sext32();
      u32 value;
      if (!ReadMemoryWord(addr, &value))
        return;

      GTE::WriteRegister(ZeroExtend32(static_cast<u8>(inst.i.rt.GetValue())), value);

      if constexpr (pgxp_mode >= PGXPMode::Memory)
        PGXP::CPU_LWC2(inst.bits, value, addr);
    }
    break;

    case InstructionOp::swc2:
    {
      if (!g_state.cop0_regs.sr.CE2)
      {
        Log_WarningPrintf("Coprocessor 2 not enabled");
        RaiseException(Exception::CpU);
        return;
      }

      const VirtualMemoryAddress addr = ReadReg(inst.i.rs) + inst.i.imm_sext32();
      const u32 value = GTE::ReadRegister(ZeroExtend32(static_cast<u8>(inst.i.rt.GetValue())));
      WriteMemoryWord(addr, value);

      if constexpr (pgxp_mode >= PGXPMode::Memory)
        PGXP::CPU_SWC2(inst.bits, value, addr);
    }
    break;

      // swc0/lwc0/cop1/cop3 are essentially no-ops
    case InstructionOp::cop1:
    case InstructionOp::cop3:
    case InstructionOp::lwc0:
    case InstructionOp::lwc1:
    case InstructionOp::lwc3:
    case InstructionOp::swc0:
    case InstructionOp::swc1:
    case InstructionOp::swc3:
    {
    }
    break;

      // everything else is reserved/invalid
    default:
    {
      u32 ram_value;
      if (SafeReadInstruction(g_state.current_instruction_pc, &ram_value) &&
          ram_value != g_state.current_instruction.bits)
      {
        Log_ErrorPrintf("Stale icache at 0x%08X - ICache: %08X RAM: %08X", g_state.current_instruction_pc,
                        g_state.current_instruction.bits, ram_value);
        g_state.current_instruction.bits = ram_value;
        goto restart_instruction;
      }

      RaiseException(Exception::RI);
    }
    break;
  }
}

void DispatchInterrupt()
{
  // If the instruction we're about to execute is a GTE instruction, delay dispatching the interrupt until the next
  // instruction. For some reason, if we don't do this, we end up with incorrectly sorted polygons and flickering..
  SafeReadInstruction(g_state.regs.pc, &g_state.next_instruction.bits);
  if (g_state.next_instruction.op == InstructionOp::cop2 && !g_state.next_instruction.cop.IsCommonInstruction())
    GTE::ExecuteInstruction(g_state.next_instruction.bits);

  // Interrupt raising occurs before the start of the instruction.
  tracer.isInterrupt = true;
  RaiseException(
    Cop0Registers::CAUSE::MakeValueForException(Exception::INT, g_state.next_instruction_is_branch_delay_slot,
                                                g_state.branch_was_taken, g_state.next_instruction.cop.cop_n),
    g_state.regs.pc);

  g_state.pending_ticks++;
}

void UpdateDebugDispatcherFlag()
{
  const bool has_any_breakpoints = !s_breakpoints.empty();

  // TODO: cop0 breakpoints
  const auto& dcic = g_state.cop0_regs.dcic;
  const bool has_cop0_breakpoints =
    dcic.super_master_enable_1 && dcic.super_master_enable_2 && dcic.execution_breakpoint_enable;

  const bool use_debug_dispatcher = has_any_breakpoints || has_cop0_breakpoints || s_trace_to_log;
  if (use_debug_dispatcher == g_state.use_debug_dispatcher)
    return;

  Log_DevPrintf("%s debug dispatcher", use_debug_dispatcher ? "Now using" : "No longer using");
  g_state.use_debug_dispatcher = use_debug_dispatcher;
  ForceDispatcherExit();
}

void ForceDispatcherExit()
{
  // zero the downcount so we break out and switch
  g_state.downcount = 0;
  g_state.frame_done = true;
}

bool HasAnyBreakpoints()
{
  return !s_breakpoints.empty();
}

bool HasBreakpointAtAddress(VirtualMemoryAddress address)
{
  for (const Breakpoint& bp : s_breakpoints)
  {
    if (bp.address == address)
      return true;
  }

  return false;
}

BreakpointList GetBreakpointList(bool include_auto_clear, bool include_callbacks)
{
  BreakpointList bps;
  bps.reserve(s_breakpoints.size());

  for (const Breakpoint& bp : s_breakpoints)
  {
    if (bp.callback && !include_callbacks)
      continue;
    if (bp.auto_clear && !include_auto_clear)
      continue;

    bps.push_back(bp);
  }

  return bps;
}

bool AddBreakpoint(VirtualMemoryAddress address, bool auto_clear, bool enabled)
{
  if (HasBreakpointAtAddress(address))
    return false;

  Log_InfoPrintf("Adding breakpoint at %08X, auto clear = %u", address, static_cast<unsigned>(auto_clear));

  Breakpoint bp{address, nullptr, auto_clear ? 0 : s_breakpoint_counter++, 0, auto_clear, enabled};
  s_breakpoints.push_back(std::move(bp));
  UpdateDebugDispatcherFlag();

  if (!auto_clear)
  {
    g_host_interface->ReportFormattedDebuggerMessage(
      g_host_interface->TranslateString("DebuggerMessage", "Added breakpoint at 0x%08X."), address);
  }

  return true;
}

bool AddBreakpointWithCallback(VirtualMemoryAddress address, BreakpointCallback callback)
{
  if (HasBreakpointAtAddress(address))
    return false;

  Log_InfoPrintf("Adding breakpoint with callback at %08X", address);

  Breakpoint bp{address, callback, 0, 0, false, true};
  s_breakpoints.push_back(std::move(bp));
  UpdateDebugDispatcherFlag();
  return true;
}

bool RemoveBreakpoint(VirtualMemoryAddress address)
{
  auto it = std::find_if(s_breakpoints.begin(), s_breakpoints.end(),
                         [address](const Breakpoint& bp) { return bp.address == address; });
  if (it == s_breakpoints.end())
    return false;

  g_host_interface->ReportFormattedDebuggerMessage(
    g_host_interface->TranslateString("DebuggerMessage", "Removed breakpoint at 0x%08X."), address);

  s_breakpoints.erase(it);
  UpdateDebugDispatcherFlag();

  if (address == s_last_breakpoint_check_pc)
    s_last_breakpoint_check_pc = INVALID_BREAKPOINT_PC;

  return true;
}

void ClearBreakpoints()
{
  s_breakpoints.clear();
  s_breakpoint_counter = 0;
  s_last_breakpoint_check_pc = INVALID_BREAKPOINT_PC;
  UpdateDebugDispatcherFlag();
}

bool AddStepOverBreakpoint()
{
  u32 bp_pc = g_state.regs.pc;

  Instruction inst;
  if (!SafeReadInstruction(bp_pc, &inst.bits))
    return false;

  bp_pc += sizeof(Instruction);

  if (!IsCallInstruction(inst))
  {
    g_host_interface->ReportFormattedDebuggerMessage(
      g_host_interface->TranslateString("DebuggerMessage", "0x%08X is not a call instruction."), g_state.regs.pc);
    return false;
  }

  if (!SafeReadInstruction(bp_pc, &inst.bits))
    return false;

  if (IsBranchInstruction(inst))
  {
    g_host_interface->ReportFormattedDebuggerMessage(
      g_host_interface->TranslateString("DebuggerMessage", "Can't step over double branch at 0x%08X"), g_state.regs.pc);
    return false;
  }

  // skip the delay slot
  bp_pc += sizeof(Instruction);

  g_host_interface->ReportFormattedDebuggerMessage(
    g_host_interface->TranslateString("DebuggerMessage", "Stepping over to 0x%08X."), bp_pc);

  return AddBreakpoint(bp_pc, true);
}

bool AddStepOutBreakpoint(u32 max_instructions_to_search)
{
  // find the branch-to-ra instruction.
  u32 ret_pc = g_state.regs.pc;
  for (u32 i = 0; i < max_instructions_to_search; i++)
  {
    ret_pc += sizeof(Instruction);

    Instruction inst;
    if (!SafeReadInstruction(ret_pc, &inst.bits))
    {
      g_host_interface->ReportFormattedDebuggerMessage(
        g_host_interface->TranslateString("DebuggerMessage",
                                          "Instruction read failed at %08X while searching for function end."),
        ret_pc);
      return false;
    }

    if (IsReturnInstruction(inst))
    {
      g_host_interface->ReportFormattedDebuggerMessage(
        g_host_interface->TranslateString("DebuggerMessage", "Stepping out to 0x%08X."), ret_pc);

      return AddBreakpoint(ret_pc, true);
    }
  }

  g_host_interface->ReportFormattedDebuggerMessage(
    g_host_interface->TranslateString("DebuggerMessage",
                                      "No return instruction found after %u instructions for step-out at %08X."),
    max_instructions_to_search, g_state.regs.pc);

  return false;
}

ALWAYS_INLINE_RELEASE static bool BreakpointCheck()
{
  const u32 pc = g_state.regs.pc;

  // single step - we want to break out after this instruction, so set a pending exit
  // the bp check happens just before execution, so this is fine
  if (s_single_step)
  {
    ForceDispatcherExit();
    s_single_step = false;
    s_last_breakpoint_check_pc = pc;
    return false;
  }

  if (pc == s_last_breakpoint_check_pc)
  {
    // we don't want to trigger the same breakpoint which just paused us repeatedly.
    return false;
  }

  u32 count = static_cast<u32>(s_breakpoints.size());
  for (u32 i = 0; i < count;)
  {
    Breakpoint& bp = s_breakpoints[i];
    if (!bp.enabled || bp.address != pc)
    {
      i++;
      continue;
    }

    bp.hit_count++;

    if (bp.callback)
    {
      // if callback returns false, the bp is no longer recorded
      if (!bp.callback(pc))
      {
        s_breakpoints.erase(s_breakpoints.begin() + i);
        count--;
        UpdateDebugDispatcherFlag();
      }
      else
      {
        i++;
      }
    }
    else
    {
      g_host_interface->PauseSystem(true);

      if (bp.auto_clear)
      {
        g_host_interface->ReportFormattedDebuggerMessage("Stopped execution at 0x%08X.", pc);
        s_breakpoints.erase(s_breakpoints.begin() + i);
        count--;
        UpdateDebugDispatcherFlag();
      }
      else
      {
        g_host_interface->ReportFormattedDebuggerMessage("Hit breakpoint %u at 0x%08X.", bp.number, pc);
        i++;
      }
    }
  }

  s_last_breakpoint_check_pc = pc;
  return System::IsPaused();
}

Tracer tracer;
void cpustate::update(State g_state)
{
  this->ticks = tracer.totalticks;
  this->newticks = g_state.lastticks + tracer.sumticks;
  tracer.sumticks = 0;
  this->pc = g_state.current_instruction_pc;

  if (tracer.isException) 
   this->opcode = tracer.exceptionOpcode;
  else 
    this->opcode = g_state.current_instruction.bits;

  for (int i = 0; i < 32; i++)
  {
    this->regs[i] = g_state.regs.r[i];
  }
  if (g_state.load_delay_reg != Reg::count)
    this->regs[static_cast<u8>(g_state.load_delay_reg)] = g_state.load_delay_value;

  this->regs_hi = g_state.regs.hi;
  this->regs_lo = g_state.regs.lo;

  this->sr = g_state.cop0_regs.sr.bits;
  this->cause = g_state.cop0_regs.cause.bits;

  this->irq = g_interrupt_controller.m_interrupt_status_register;

  this->gpu_time = g_gpu->m_crtc_tick_event->m_downcount;
  this->gpu_line = g_gpu->m_crtc_state.current_scanline;
  this->gpu_stat = g_gpu->m_GPUSTAT.bits;
  this->fifocount = g_gpu->m_fifo.GetSize();
  this->gpu_ticks = g_gpu->m_pending_command_ticks_last;

  this->mdec_stat = g_mdec.m_status.bits;

  this->cd_status = g_cdrom.m_status.bits | (g_cdrom.m_secondary_status.bits << 8);

  for (int i = 0; i < 3; i++)
  {
      this->timer[i] = g_timers.m_states[i].counter;
  }

  //this->debug8 = Bus::g_ram[0x001ffe98];
  //this->debug8 = g_gpu->m_crtc_state.interlaced_field;
  //this->debug8 = g_dma.m_state[4].request;
  this->debug8 = g_spu.m_transfer_fifo.GetSize();
  //this->debug16 = g_gpu->m_vram_ptr[0x7FFFC];
  this->debug16 = g_spu.m_SPUSTAT.bits;
  //this->debug32 = *(uint32_t*)&Bus::g_ram[0x7abd0];
  //this->debug32 = *(uint32_t*)&g_state.dcache[0x08];
  //this->debug32 = g_cdrom.m_command_event->m_downcount;
  //this->debug32 = g_gpu->m_command_tick_event->m_downcount;
  this->debug32 = g_spu.m_transfer_event->m_downcount;
  //this->debug32 = g_mdec.m_block_copy_out_event->m_downcount;
  //this->debug32 = g_gpu->m_crtc_state.vertical_display_end;
  //this->debug32 = g_gpu->m_GPUREAD_latch;
}

inline void printsingle(FILE* file, uint32_t value, const char * name, int size)
{
  char buf[10];
  fputs(name, file);
  fputc(' ', file);
  _itoa(value, buf, 16);
  for (int c = strlen(buf); c < size; c++) fputc('0', file);
  fputs(buf, file);
  //fputc('\n', file);
  fputc(' ', file);
}

inline void printchange(FILE* file, uint32_t oldvalue, uint32_t newvalue, const char* name, int size)
{
  char buf[10];
  fputs(name, file);
  fputc(' ', file);
  //_itoa(oldvalue, buf, 16);
  //for (int c = strlen(buf); c < size; c++) fputc('0', file);
  //fputs(buf, file);
  //fputc(' ', file);
  _itoa(newvalue, buf, 16);
  for (int c = strlen(buf); c < size; c++) fputc('0', file);
  fputs(buf, file);
  //fputc('\n', file);
  fputc(' ', file);
}

void Tracer::VramOutWriteFile()
{
#ifdef VRAMFILEOUT
    FILE* file = fopen("R:\\debug_gpu_duck.txt", "w");

    for (int i = 0; i < tracer.debug_VramOutCount; i++)
    {
        if (debug_VramOutType[i] == 1) fputs("Pixel: ", file);
        if (debug_VramOutType[i] == 2) fputs("Fifo: ", file);
        if (debug_VramOutType[i] == 3) fputs("LinkedList: ", file);
        char buf[10];
        _itoa(tracer.debug_VramOutTime[i], buf, 16);
        for (int c = strlen(buf); c < 8; c++) fputc('0', file);
        fputs(buf, file);
        fputc(' ', file);
        _itoa(tracer.debug_VramOutAddr[i], buf, 16);
        for (int c = strlen(buf); c < 8; c++) fputc('0', file);
        fputs(buf, file);
        fputc(' ', file);
        _itoa(tracer.debug_VramOutData[i], buf, 16);
        for (int c = strlen(buf); c < 4; c++) fputc('0', file);
        fputs(buf, file);
    
        fputc('\n', file);
    }
    fclose(file);

    file = fopen("R:\\debug_pixel_duck.txt", "w");

    for (int i = 0; i < debug_VramOutCount; i++)
    {
        if (debug_VramOutType[i] == 1)
        {
            char buf[10];
            uint16_t x = (debug_VramOutAddr[i] >> 1) & 0x3FF;
            uint16_t y = (debug_VramOutAddr[i] >> 11) & 0x1FF;
            _itoa(x, buf, 10);
            fputs(buf, file);
            fputc(' ', file);
            _itoa(y, buf, 10);
            fputs(buf, file);
            fputc(' ', file);
            _itoa(debug_VramOutData[i], buf, 10);
            fputs(buf, file);
            fputc('\n', file);
        }
    }
    fclose(file);
#endif
}

void Tracer::GTEoutWriteFile()
{
#ifdef GTEFILEOUT
    FILE* file = fopen("R:\\debug_gte_duck.txt", "w");

    for (int i = 0; i < tracer.debug_GTEOutCount; i++)
    {
        if (debug_GTEOutType[i] == 1) fputs("COMMAND: ", file);
        if (debug_GTEOutType[i] == 2) fputs("WRITE REG: ", file);
        if (debug_GTEOutType[i] == 3) fputs("COMMAND REG: ", file);
        char buf[10];
        _itoa(tracer.debug_GTEOutTime[i], buf, 16);
        for (int c = strlen(buf); c < 8; c++) fputc('0', file);
        fputs(buf, file);
        fputc(' ', file);
        _itoa(tracer.debug_GTEOutAddr[i], buf, 10);
        for (int c = strlen(buf); c < 2; c++) fputc('0', file);
        fputs(buf, file);
        fputc(' ', file);
        _itoa(tracer.debug_GTEOutData[i], buf, 16);
        for (int c = strlen(buf); c < 8; c++) fputc('0', file);
        fputs(buf, file);

        fputc('\n', file);
    }
    fclose(file);

    file = fopen("R:\\gte_test_duck.txt", "w");

    for (int i = 0; i < tracer.debug_GTEOutCount; i++)
    {
        if (tracer.debug_GTEOutType[i] < 3)
        {
            char buf[10];
            _itoa(tracer.debug_GTEOutType[i], buf, 16);
            for (int c = strlen(buf); c < 2; c++) fputc('0', file);
            fputs(buf, file);
            fputc(' ', file);
            _itoa(tracer.debug_GTEOutAddr[i], buf, 16);
            for (int c = strlen(buf); c < 2; c++) fputc('0', file);
            fputs(buf, file);
            fputc(' ', file);
            _itoa(tracer.debug_GTEOutData[i], buf, 16);
            for (int c = strlen(buf); c < 8; c++) fputc('0', file);
            fputs(buf, file);

            fputc('\n', file);
        }
    }
    fclose(file);
#endif
}

void Tracer::GTEoutRegCapture(uint8_t regtype)
{
#ifdef GTEFILEOUT
    if (debug_GTEOutCount >= 1000000 - 64) return;

    for (int i = 0; i < 64; i++)
    {
        if (debug_GTELast[i] != g_state.gte_regs.r32[i])
        {
            CPU::tracer.debug_GTEOutTime[CPU::tracer.debug_GTEOutCount] = tracer.commands;
            CPU::tracer.debug_GTEOutAddr[CPU::tracer.debug_GTEOutCount] = i;
            CPU::tracer.debug_GTEOutData[CPU::tracer.debug_GTEOutCount] = g_state.gte_regs.r32[i];
            CPU::tracer.debug_GTEOutType[CPU::tracer.debug_GTEOutCount] = regtype;
            CPU::tracer.debug_GTEOutCount++;
            debug_GTELast[i] = g_state.gte_regs.r32[i];
        }
    }
#endif
}

void Tracer::GTEoutCommandCapture(uint32_t command)
{
#ifdef GTEFILEOUT
    if (debug_GTEOutCount >= 1000000) return;

    CPU::tracer.debug_GTEOutTime[CPU::tracer.debug_GTEOutCount] = tracer.commands;
    CPU::tracer.debug_GTEOutAddr[CPU::tracer.debug_GTEOutCount] = 0;
    CPU::tracer.debug_GTEOutData[CPU::tracer.debug_GTEOutCount] = command;
    CPU::tracer.debug_GTEOutType[CPU::tracer.debug_GTEOutCount] = 1;
    CPU::tracer.debug_GTEOutCount++;
#endif
}

void Tracer::GTETest()
{
    std::ifstream infile("R:\\gte_test_duck.txt");
    std::string line;
    while (std::getline(infile, line))
    {
        std::string type = line.substr(0, 2);
        std::string addr = line.substr(3, 2);
        std::string data = line.substr(6, 8);
        u8 typeI = std::stoul(type, nullptr, 16);
        u8 addrI = std::stoul(addr, nullptr, 16);
        u32 dataI = std::stoul(data, nullptr, 16);
        switch (typeI)
        {
        case 1:  GTE::ExecuteInstruction(dataI); break;
        case 2:  GTE::WriteRegister(addrI, dataI); break;
        }
    }

    GTEoutWriteFile();
}

void Tracer::MDECOutCapture(uint8_t type, uint8_t addr, uint32_t data)
{
#ifdef MDECFILEOUT
    if (debug_MDECOutCount == 694758)
    {
        int a = 5;
    }

    if (CPU::tracer.debug_MDECOutCount >= 1000000) return;

    //CPU::tracer.debug_MDECOutTime[CPU::tracer.debug_MDECOutCount] = totalticks + CPU::g_state.pending_ticks - 1;
    CPU::tracer.debug_MDECOutTime[CPU::tracer.debug_MDECOutCount] = totalticks;
    CPU::tracer.debug_MDECOutAddr[CPU::tracer.debug_MDECOutCount] = addr;
    CPU::tracer.debug_MDECOutData[CPU::tracer.debug_MDECOutCount] = data;
    CPU::tracer.debug_MDECOutType[CPU::tracer.debug_MDECOutCount] = type;
    CPU::tracer.debug_MDECOutCount++;
#endif
}

void Tracer::MDECOutWriteFile(bool writeTest)
{
#ifdef MDECFILEOUT
    FILE* file = fopen("R:\\debug_mdec_duck.txt", "w");

    for (int i = 0; i < tracer.debug_MDECOutCount; i++)
    {
        if (debug_MDECOutType[i] == 1) fputs("Pixel: ", file);
        if (debug_MDECOutType[i] == 2) fputs("Fifo: ", file);
        if (debug_MDECOutType[i] == 3) fputs("Blockendpos: ", file);
        if (debug_MDECOutType[i] == 4) fputs("Blockresult: ", file);
        if (debug_MDECOutType[i] == 5) fputs("IDCTresult: ", file);
        if (debug_MDECOutType[i] == 6) fputs("FIFOLeft: ", file);
        if (debug_MDECOutType[i] == 7) fputs("CPUREAD: ", file);
        if (debug_MDECOutType[i] == 8) fputs("CPUWRITE: ", file);
        if (debug_MDECOutType[i] == 9) fputs("DMAREAD: ", file);
        if (debug_MDECOutType[i] == 10) fputs("DMAWRITE: ", file);
        if (debug_MDECOutType[i] == 11) fputs("EVENT: ", file);
        char buf[10];
        _itoa(tracer.debug_MDECOutTime[i], buf, 16);
        for (int c = strlen(buf); c < 8; c++) fputc('0', file);
        fputs(buf, file);
        fputc(' ', file);
        _itoa(tracer.debug_MDECOutAddr[i], buf, 16);
        for (int c = strlen(buf); c < 2; c++) fputc('0', file);
        fputs(buf, file);
        fputc(' ', file);
        _itoa(tracer.debug_MDECOutData[i], buf, 16);
        for (int c = strlen(buf); c < 8; c++) fputc('0', file);
        fputs(buf, file);

        fputc('\n', file);
    }
    fclose(file);

    if (writeTest)
    {
        file = fopen("R:\\mdec_test_duck.txt", "w");

        for (int i = 0; i < tracer.debug_MDECOutCount; i++)
        {
            if (debug_MDECOutType[i] >= 7)
            {
                char buf[10];
                _itoa(tracer.debug_MDECOutType[i], buf, 16);
                for (int c = strlen(buf); c < 2; c++) fputc('0', file);
                fputs(buf, file);
                fputc(' ', file);
                _itoa(tracer.debug_MDECOutTime[i], buf, 16);
                for (int c = strlen(buf); c < 8; c++) fputc('0', file);
                fputs(buf, file);
                fputc(' ', file);
                _itoa(tracer.debug_MDECOutAddr[i], buf, 16);
                for (int c = strlen(buf); c < 2; c++) fputc('0', file);
                fputs(buf, file);
                fputc(' ', file);
                _itoa(tracer.debug_MDECOutData[i], buf, 16);
                for (int c = strlen(buf); c < 8; c++) fputc('0', file);
                fputs(buf, file);

                fputc('\n', file);
            }
        }
        fclose(file);
    }
#endif
}

void Tracer::MDECTest()
{
    std::ifstream infile("R:\\mdec_test_duck.txt");
    std::string line;
    u32 timer = 0;
    //CPU::g_state.pending_ticks = 1;
    while (std::getline(infile, line))
    {
        std::string type = line.substr(0, 2);
        std::string time = line.substr(3, 8);
        std::string addr = line.substr(12, 2);
        std::string data = line.substr(15, 8);
        u8 typeI = std::stoul(type, nullptr, 16);
        u32 timeI = std::stoul(time, nullptr, 16);
        u8 addrI = std::stoul(addr, nullptr, 16);
        u32 dataI = std::stoul(data, nullptr, 16);
        while (timer < timeI)
        {
            timer++;
            totalticks++;
            if (g_mdec.m_block_copy_out_event->m_active)
            {
                g_mdec.m_block_copy_out_event->m_downcount--;
                if (g_mdec.m_block_copy_out_event->m_downcount <= 0)
                {
                    g_mdec.m_block_copy_out_event->m_downcount += g_mdec.m_block_copy_out_event->m_interval;
                    g_mdec.CopyOutBlock();
                }
            }
        }
        switch (typeI)
        {
        case 7:  g_mdec.ReadRegister(addrI); break;
        case 8:  g_mdec.WriteRegister(addrI, dataI); break;

        case 9:
        {
            u32* dest_pointer = reinterpret_cast<u32*>(&Bus::g_ram[0]);
            g_mdec.DMARead(dest_pointer, dataI);
            if (g_mdec.m_data_out_fifo.IsEmpty())
            {
                g_mdec.Execute();
            }
        }
        break;

        case 10:
            CPU::tracer.MDECOutCapture(10, 0, dataI);
            g_mdec.WriteCommandRegister(dataI);
            break;

        case 11:
            if (g_mdec.m_block_copy_out_event->m_active)
            {
                g_mdec.CopyOutBlock();
            }
        }
    }

    MDECOutWriteFile(false);
}

void Tracer::CDOutCapture(uint8_t type, uint8_t addr, uint32_t data)
{
#ifdef CDFILEOUT
    if (debug_CDOutCount == 712562)
    {
        int a = 5;
    }

    if (debug_CDOutCount >= CDFILEOUTMAX) return;

    debug_CDOutTime[debug_CDOutCount] = totalticks;
    debug_CDOutAddr[debug_CDOutCount] = addr;
    debug_CDOutData[debug_CDOutCount] = data;
    debug_CDOutType[debug_CDOutCount] = type;
    debug_CDOutCount++;
#endif
}

void Tracer::XAOutCapture(uint8_t type, uint32_t data)
{
#ifdef CDFILEOUT
    if (debug_XAOutCount >= 1000000) return;

    if (data != 0)
    {
        int a = 5;
    }

    debug_XAOutType[debug_XAOutCount] = type;
    debug_XAOutData[debug_XAOutCount] = data;
    debug_XAOutCount++;
#endif
}

void Tracer::CDOutWriteFile(bool writeTest)
{
#ifdef CDFILEOUT
    FILE* file = fopen("R:\\debug_cd_duck.txt", "w");

    for (int i = 0; i < tracer.debug_CDOutCount; i++)
    {
        if (debug_CDOutType[i] == 1) fputs("CMD: ", file);
        if (debug_CDOutType[i] == 2) fputs("DATA: ", file);
        if (debug_CDOutType[i] == 3) fputs("RSPFIFO: ", file);
        if (debug_CDOutType[i] == 4) fputs("RSPFIFO2: ", file);
        if (debug_CDOutType[i] == 5) fputs("RSPERROR: ", file);
        if (debug_CDOutType[i] == 7) fputs("WPTR: ", file);
        if (debug_CDOutType[i] == 8) fputs("CPUREAD: ", file);
        if (debug_CDOutType[i] == 9) fputs("CPUWRITE: ", file);
        if (debug_CDOutType[i] == 10) fputs("DMAREAD: ", file);
        if (debug_CDOutType[i] == 11) fputs("CDDAOUT: ", file);
        if (debug_CDOutType[i] == 12) fputs("SECTORREAD: ", file);
        char buf[10];
        _itoa(tracer.debug_CDOutTime[i], buf, 16);
        for (int c = strlen(buf); c < 8; c++) fputc('0', file);
        fputs(buf, file);
        fputc(' ', file);
        _itoa(tracer.debug_CDOutAddr[i], buf, 16);
        for (int c = strlen(buf); c < 2; c++) fputc('0', file);
        fputs(buf, file);
        fputc(' ', file);
        _itoa(tracer.debug_CDOutData[i], buf, 16);
        for (int c = strlen(buf); c < 8; c++) fputc('0', file);
        fputs(buf, file);

        fputc('\n', file);
    }
    fclose(file);

    file = fopen("R:\\debug_xa_duck.txt", "w");
    for (int i = 0; i < tracer.debug_XAOutCount; i++)
    {
        if (debug_XAOutType[i] == 1) fputs("DATAIN: ", file);
        if (debug_XAOutType[i] == 2) fputs("ADPCMCALC: ", file);
        if (debug_XAOutType[i] == 3) fputs("ADPCMOUT: ", file);
        if (debug_XAOutType[i] == 4) fputs("OUT: ", file);
        char buf[10];
        _itoa(i, buf, 16);
        for (int c = strlen(buf); c < 6; c++) fputc('0', file);
        fputs(buf, file);
        fputc(' ', file);
        _itoa(tracer.debug_XAOutData[i], buf, 16);
        for (int c = strlen(buf); c < 8; c++) fputc('0', file);
        fputs(buf, file);

        fputc('\n', file);
    }
    fclose(file);

    if (writeTest)
    {
        file = fopen("R:\\cd_test_duck.txt", "w");

        for (int i = 0; i < tracer.debug_CDOutCount; i++)
        {
            if (debug_CDOutType[i] >= 8)
            {
                char buf[10];
                _itoa(tracer.debug_CDOutType[i], buf, 16);
                for (int c = strlen(buf); c < 2; c++) fputc('0', file);
                fputs(buf, file);
                fputc(' ', file);
                _itoa(tracer.debug_CDOutTime[i], buf, 16);
                for (int c = strlen(buf); c < 8; c++) fputc('0', file);
                fputs(buf, file);
                fputc(' ', file);
                _itoa(tracer.debug_CDOutAddr[i], buf, 16);
                for (int c = strlen(buf); c < 2; c++) fputc('0', file);
                fputs(buf, file);
                fputc(' ', file);
                _itoa(tracer.debug_CDOutData[i], buf, 16);
                for (int c = strlen(buf); c < 8; c++) fputc('0', file);
                fputs(buf, file);

                fputc('\n', file);
            }
        }
        fclose(file);
    }
#endif
}

void Tracer::CDTest()
{
    std::ifstream infile("R:\\cd_test_duck.txt");
    std::string line;
    u32 timer = 0;
    while (std::getline(infile, line))
    {
        std::string type = line.substr(0, 2);
        std::string time = line.substr(3, 8);
        std::string addr = line.substr(12, 2);
        std::string data = line.substr(15, 8);
        u8 typeI = std::stoul(type, nullptr, 16);
        u32 timeI = std::stoul(time, nullptr, 16);
        u8 addrI = std::stoul(addr, nullptr, 16);
        u32 dataI = std::stoul(data, nullptr, 16);
        while (timer < timeI)
        {
            timer++;
            totalticks++;
            TimingEvents::AddGlobalTickCounter();
            if (g_cdrom.m_command_event->m_active)
            {
                g_cdrom.m_command_event->m_downcount--;
                if (g_cdrom.m_command_event->m_downcount <= 0)
                {
                    g_cdrom.m_command_event->m_downcount += g_cdrom.m_command_event->m_interval;
                    g_cdrom.ExecuteCommand(0);
                }
            }
            if (g_cdrom.m_command_second_response_event->m_active)
            {
                g_cdrom.m_command_second_response_event->m_downcount--;
                if (g_cdrom.m_command_second_response_event->m_downcount <= 0)
                {
                    g_cdrom.m_command_second_response_event->m_downcount += g_cdrom.m_command_second_response_event->m_interval;
                    g_cdrom.ExecuteCommandSecondResponse(0);
                }
            }
            if (g_cdrom.m_drive_event->m_active)
            {
                g_cdrom.m_drive_event->m_downcount--;
                if (g_cdrom.m_drive_event->m_downcount <= 0)
                {
                    g_cdrom.m_drive_event->m_downcount += g_cdrom.m_drive_event->m_interval;
                    g_cdrom.ExecuteDrive(0);
                }
            }
        }

        switch (typeI)
        {
        case 8:  g_cdrom.ReadRegister(addrI); break;
        case 9:  g_cdrom.WriteRegister(addrI, dataI); break;
        case 10:
        {
            u32* dest_pointer = reinterpret_cast<u32*>(&Bus::g_ram[0]);
            g_cdrom.DMARead(dest_pointer, dataI);
        }
            break;
        }
    }

    CDOutWriteFile(false);
}

void Tracer::PadOutCapture(uint8_t addr, uint16_t data, uint8_t type)
{
    if (type < 8) return;

#ifdef PADFILEOUT
    if (debug_PadOutCount >= 1000000) return;

    debug_PadOutTime[debug_PadOutCount] = totalticks;
    debug_PadOutAddr[debug_PadOutCount] = addr;
    debug_PadOutData[debug_PadOutCount] = data;
    debug_PadOutType[debug_PadOutCount] = type;
    debug_PadOutCount++;
#endif
}

void Tracer::PadOutWriteFile(bool writeTest)
{
#ifdef PADFILEOUT
    FILE* file = fopen("R:\\debug_pad_duck.txt", "w");

    for (int i = 0; i < tracer.debug_PadOutCount; i++)
    {
        if (debug_PadOutType[i] == 1) fputs("WRITE: ", file);
        if (debug_PadOutType[i] == 2) fputs("READ: ", file);
        if (debug_PadOutType[i] == 3) fputs("TRANSMIT: ", file);
        if (debug_PadOutType[i] == 4) fputs("IRQ: ", file);
        if (debug_PadOutType[i] == 5) fputs("BEGINTRANSFER: ", file);
        if (debug_PadOutType[i] == 6) fputs("READMEMBLOCK: ", file);
        if (debug_PadOutType[i] == 7) fputs("READMEMDATA: ", file);
        if (debug_PadOutType[i] == 8) fputs("RESETCONTROLLER: ", file);
        if (debug_PadOutType[i] == 9) fputs("TRANSFER: ", file);
        char buf[10];
        //_itoa(tracer.debug_PadOutTime[i], buf, 16);
        //for (int c = strlen(buf); c < 8; c++) fputc('0', file);
        //fputs(buf, file);
        //fputc(' ', file);
        _itoa(tracer.debug_PadOutAddr[i], buf, 16);
        for (int c = strlen(buf); c < 2; c++) fputc('0', file);
        fputs(buf, file);
        fputc(' ', file);
        _itoa(tracer.debug_PadOutData[i], buf, 16);
        for (int c = strlen(buf); c < 4; c++) fputc('0', file);
        fputs(buf, file);

        fputc('\n', file);
    }
    fclose(file);

    if (writeTest)
    {
        file = fopen("R:\\pad_test_duck.txt", "w");

        for (int i = 0; i < tracer.debug_PadOutCount; i++)
        {
            if (debug_PadOutType[i] < 8) continue;
            char buf[10];
            _itoa(tracer.debug_PadOutType[i], buf, 16);
            for (int c = strlen(buf); c < 2; c++) fputc('0', file);
            fputs(buf, file);
            fputc(' ', file);
            _itoa(tracer.debug_PadOutAddr[i], buf, 16);
            for (int c = strlen(buf); c < 2; c++) fputc('0', file);
            fputs(buf, file);
            fputc(' ', file);
            _itoa(tracer.debug_PadOutData[i], buf, 16);
            for (int c = strlen(buf); c < 2; c++) fputc('0', file);
            fputs(buf, file);

            fputc('\n', file);
        }
        fclose(file);
    }
#endif
}

void Tracer::PadTest()
{
#ifdef PADFILEOUT
    tracer.debug_PadOutCount = 0;
#endif
    std::ifstream infile("R:\\pad_test_duck.txt");
    std::string line;
    u32 timer = 0;
    while (std::getline(infile, line))
    {
        std::string type = line.substr(0, 2);
        std::string addr = line.substr(3, 2);
        std::string data = line.substr(6, 2);
        u8 typeI = std::stoul(type, nullptr, 16);
        u8 addrI = std::stoul(addr, nullptr, 16);
        u8 dataI = std::stoul(data, nullptr, 16);
       
        switch (typeI)
        {

        case 8: g_pad.ResetDeviceTransferState(); break;
        case 9:
        {
            u8 data_out = addrI;
            u8 data_in = 0xFF;
            g_pad.m_controllers[0]->Transfer(data_out, &data_in);
            PadOutCapture(data_out, data_in, 9);
        }
        break;
        }
    }

    PadOutWriteFile(false);
}

void Tracer::SPUOutCapture(uint16_t addr, uint16_t data, uint8_t type, uint8_t timeadd)
{
#ifdef SPUFILEOUT
    if (debug_SPUOutCount == 2789427)
    {
        int a = 5;
    }

    if (debug_SPUOutCount >= SPUFILEOUTMAX) 
        return;

    if (!debug_SPUOutAll && type > 4) return;

    debug_SPUOutTime[debug_SPUOutCount] = totalticks + timeadd;
    debug_SPUOutAddr[debug_SPUOutCount] = addr;
    debug_SPUOutData[debug_SPUOutCount] = data;
    debug_SPUOutType[debug_SPUOutCount] = type;
    debug_SPUOutCount++;
#endif
}

void Tracer::SPUOutWriteFile(bool writeTest)
{
#ifdef SPUFILEOUT
    FILE* file = fopen("R:\\debug_sound_duck.txt", "w");

    for (int i = 0; i < debug_SPUOutCount; i++)
    {
        if (debug_SPUOutType[i] == 1) fputs("WRITEREG: ", file);
        if (debug_SPUOutType[i] == 2) fputs("READREG: ", file);
        if (debug_SPUOutType[i] == 3) fputs("DMAWRITE: ", file);
        if (debug_SPUOutType[i] == 4) fputs("DMAREAD: ", file);
        if (debug_SPUOutType[i] == 5) fputs("SAMPLEOUT: ", file);
        if (debug_SPUOutType[i] == 6) fputs("ADPCM: ", file);
        if (debug_SPUOutType[i] == 7) fputs("CHAN: ", file);
        if (debug_SPUOutType[i] == 8) fputs("ADSRTICKS: ", file);
        if (debug_SPUOutType[i] == 9) fputs("REVERBWRITE: ", file);
        if (debug_SPUOutType[i] == 10) fputs("REVERBREAD: ", file);
        if (debug_SPUOutType[i] == 11) fputs("REVERBSAMPLE: ", file);
        if (debug_SPUOutType[i] == 12) fputs("CAPTURE: ", file);
        if (debug_SPUOutType[i] == 13) fputs("ENVCHAN: ", file);
        if (debug_SPUOutType[i] == 14) fputs("NOISE: ", file);
        if (debug_SPUOutType[i] == 15) fputs("DMARAM: ", file);
        if (debug_SPUOutType[i] == 16) fputs("ADSRVOLUME: ", file);
        if (debug_SPUOutType[i] == 17) fputs("IRQ: ", file);
        if (debug_SPUOutType[i] == 18) fputs("VOICEADDRESS: ", file);
        char buf[10];
        _itoa(debug_SPUOutTime[i], buf, 16);
        for (int c = strlen(buf); c < 8; c++) fputc('0', file);
        fputs(buf, file);
        fputc(' ', file);
        _itoa(debug_SPUOutAddr[i], buf, 16);
        for (int c = strlen(buf); c < 4; c++) fputc('0', file);
        fputs(buf, file);
        fputc(' ', file);
        _itoa(debug_SPUOutData[i], buf, 16);
        for (int c = strlen(buf); c < 4; c++) fputc('0', file);
        fputs(buf, file);

        fputc('\n', file);
    }
    fclose(file);

    if (writeTest)
    {
        file = fopen("R:\\sound_test_duck.txt", "w");

        for (int i = 0; i < debug_SPUOutCount; i++)
        {
            if (debug_SPUOutType[i] <= 4)
            {
                char buf[10];
                _itoa(debug_SPUOutType[i], buf, 16);
                for (int c = strlen(buf); c < 2; c++) fputc('0', file);
                fputs(buf, file);
                fputc(' ', file);
                _itoa(debug_SPUOutTime[i], buf, 16);
                for (int c = strlen(buf); c < 8; c++) fputc('0', file);
                fputs(buf, file);
                fputc(' ', file);
                _itoa(debug_SPUOutAddr[i], buf, 16);
                for (int c = strlen(buf); c < 4; c++) fputc('0', file);
                fputs(buf, file);
                fputc(' ', file);
                _itoa(debug_SPUOutData[i], buf, 16);
                for (int c = strlen(buf); c < 4; c++) fputc('0', file);
                fputs(buf, file);

                fputc('\n', file);
            }
        }
        fclose(file);
    }
#endif
}

void Tracer::SPUTest()
{
    std::ifstream infile("R:\\sound_test_duck.txt");
    std::string line;
    u32 timer = 0;
    u32 dmaTicks = 0;
    int cmdcnt = 0;
    g_spu.m_tick_event->m_time_since_last_run = 1;
    g_spu.m_tick_event->m_downcount = 767;

    g_spu.m_ram.fill(0);
    for (u32 i = 0; i < g_spu.NUM_VOICES; i++)
    {
        g_spu.m_voices[i].current_address = 0;
        std::fill_n(g_spu.m_voices[i].regs.index, g_spu.NUM_VOICE_REGISTERS, u16(0));
        g_spu.m_voices[i].counter.bits = 0;
        g_spu.m_voices[i].current_block_flags.bits = 0;
        g_spu.m_voices[i].is_first_block = 0;
        g_spu.m_voices[i].current_block_samples.fill(s16(0));
        g_spu.m_voices[i].adpcm_last_samples.fill(s32(0));
        g_spu.m_voices[i].adsr_envelope.Reset(0, false, false);
        g_spu.m_voices[i].adsr_phase = SPU::ADSRPhase::Off;
        g_spu.m_voices[i].adsr_target = 0;
        g_spu.m_voices[i].has_samples = false;
        g_spu.m_voices[i].ignore_loop_address = false;
    }

    while (std::getline(infile, line))
    {
        std::string type = line.substr(0, 2);
        std::string time = line.substr(3, 8);
        std::string addr = line.substr(12, 4);
        std::string data = line.substr(17, 4);
        u8 typeI = std::stoul(type, nullptr, 16);
        u32 timeI = std::stoul(time, nullptr, 16);
        u16 addrI = std::stoul(addr, nullptr, 16);
        u16 dataI = std::stoul(data, nullptr, 16);
        while (timer < timeI)
        {
            timer++;
            totalticks++;
            TimingEvents::AddGlobalTickCounter();
            g_spu.m_tick_event->m_time_since_last_run++;
            g_spu.m_transfer_event->m_time_since_last_run++;
            if (g_spu.m_tick_event->m_active)
            {
                g_spu.m_tick_event->m_downcount--;
                if (g_spu.m_tick_event->m_downcount <= 0)
                {
                    g_spu.m_tick_event->m_downcount += g_spu.m_tick_event->m_interval;
                    g_spu.Execute(g_spu.m_tick_event->m_time_since_last_run);
                    g_spu.m_tick_event->m_time_since_last_run = 0;
                }
            }
            if (g_spu.m_transfer_event->m_active)
            {
                g_spu.m_transfer_event->m_downcount--;
                if (g_spu.m_transfer_event->m_downcount <= 0)
                {
                    g_spu.m_transfer_event->m_downcount += g_spu.m_transfer_event->m_interval;
                    g_spu.ExecuteTransfer(g_spu.m_transfer_event->m_time_since_last_run);
                    g_spu.m_transfer_event->m_time_since_last_run = 0;
                }
            }
        }

        if (timer == 0x1ec7460a)
        {
            int a = 5;
        }

        switch (typeI)
        {
        case 1: g_spu.WriteRegister(addrI, dataI); break;
        case 2: g_spu.ReadRegister(addrI); break;
        case 3: 
            g_spu.m_transfer_fifo.Push(dataI); 
            SPUOutCapture(addrI, dataI, 3, 0);
            if (addrI == 1)
            {
                dmaTicks = 0;
            }
            else dmaTicks++;
            if (addrI == 3)
            {
                g_spu.UpdateDMARequest();
                g_spu.UpdateTransferEvent();
                g_spu.m_transfer_event->m_downcount -= dmaTicks;
                g_spu.m_transfer_event->m_time_since_last_run += dmaTicks;
            }
            break;
        case 4: 
            u16 retval = g_spu.m_transfer_fifo.Pop();
            SPUOutCapture(addrI, retval, 4, 0);
            if (addrI == 1) dmaTicks = 0; else dmaTicks++;
            if (addrI == 3)
            {
                g_spu.UpdateDMARequest();
                g_spu.UpdateTransferEvent();
                g_spu.m_transfer_event->m_downcount -= dmaTicks;
                g_spu.m_transfer_event->m_time_since_last_run += dmaTicks;
            }
            break;
        break;
        }
        cmdcnt++;
    }

    SPUOutWriteFile(false);
}

void Tracer::trace_file_last()
{
    FILE* file;
    if (tracer.debug_outdiv == 1) file = fopen("R:\\debug_duck.txt", "w");
    else file = fopen("R:\\debug_duck_n.txt", "w");

    for (int i = 0; i < traclist_ptr; i++)
    {
        cpustate laststate = Tracelist[i];
        cpustate state = Tracelist[i];
        if (i > 0)
        {
          laststate = Tracelist[i - 1];
        }

        printsingle(file, state.ticks, "#", 8);
        printsingle(file, state.newticks, "#", 3);
        printsingle(file, state.pc, "PC", 8);
        printsingle(file, state.opcode, "OP", 8);

        for (int j = 0; j < 32; j++)
        {
          if (i == 0 || state.regs[j] != laststate.regs[j])
          {
            fputc('R', file);
            char buf[10];
            _itoa(j, buf, 10);
            if (j < 10) fputc('0', file);
            printchange(file, laststate.regs[j], state.regs[j], buf, 8);
          }
        }

        //if (i == 0 || state.regs_hi != laststate.regs_hi) printchange(file, laststate.regs_hi, state.regs_hi, "HI", 8);
        //if (i == 0 || state.regs_lo != laststate.regs_lo) printchange(file, laststate.regs_lo, state.regs_lo, "LO", 8);

        //if (i == 0 || state.sr != laststate.sr) printchange(file, laststate.sr, state.sr, "SR", 8);
        if (i == 0 || state.cause != laststate.cause) printchange(file, laststate.cause, state.cause, "CAUSE", 8);

        if (i == 0 || state.irq != laststate.irq) printchange(file, laststate.irq, state.irq, "IRQ", 4);

        if (i == 0 || state.gpu_time != laststate.gpu_time) printchange(file, laststate.gpu_time, state.gpu_time, "GTM", 3);
        if (i == 0 || state.gpu_line != laststate.gpu_line) printchange(file, laststate.gpu_line, state.gpu_line, "LINE", 3);
        if (i == 0 || state.gpu_stat != laststate.gpu_stat) printchange(file, laststate.gpu_stat, state.gpu_stat, "GPUS", 8);
        if (i == 0 || state.fifocount != laststate.fifocount) printchange(file, laststate.fifocount, state.fifocount, "FIFO", 4);
        if (i == 0 || state.gpu_ticks > 0) printchange(file, laststate.gpu_ticks, state.gpu_ticks, "GTCK", 4);

        if (i == 0 || state.mdec_stat != laststate.mdec_stat) printchange(file, laststate.mdec_stat, state.mdec_stat, "MDEC", 8);

        if (i == 0 || state.cd_status != laststate.cd_status) printchange(file, laststate.cd_status, state.cd_status, "CDS", 4);

        if (i == 0 || state.timer[0] != laststate.timer[0]) printchange(file, laststate.timer[0], state.timer[0], "T0", 4);
        if (i == 0 || state.timer[1] != laststate.timer[1]) printchange(file, laststate.timer[1], state.timer[1], "T1", 4);
        if (i == 0 || state.timer[2] != laststate.timer[2]) printchange(file, laststate.timer[2], state.timer[2], "T2", 4);

        if (i == 0 || state.debug8 != laststate.debug8) printchange(file, laststate.debug8, state.debug8, "D8", 2);
        if (i == 0 || state.debug16 != laststate.debug16) printchange(file, laststate.debug16, state.debug16, "D16", 4);
        if (i == 0 || state.debug32 != laststate.debug32) printchange(file, laststate.debug32, state.debug32, "D32", 8);

        fputc('\n', file);
    }
    fclose(file);
}

void trace(State g_state)
{
    if (tracer.commands == 000000 && tracer.runmoretrace == -1)
    {
        //tracer.forceanalog = true;
        //g_cdrom.RemoveMedia(false, true);

        //tracer.GTETest();
        //tracer.CDTest();
        //tracer.MDECTest();
        tracer.debug_SPUOutAll= true;
        tracer.SPUTest();
        //tracer.PadTest();
        tracer.traclist_ptr = 0;
        tracer.runmoretrace = 2014824448;
        tracer.debug_outdiv = 1;
        FILE* file = fopen("R:\\debug_duck_tty.txt", "w");
        fclose(file);
    }

    if (tracer.runmoretrace > 0 && tracer.debug_outdivcnt == 0)
    {
        tracer.totalticks += g_state.lastticks;
        if (tracer.traclist_ptr < tracer.Tracelist_Length) tracer.Tracelist[tracer.traclist_ptr].update(g_state);
        tracer.traclist_ptr++;
        tracer.runmoretrace = tracer.runmoretrace - 1;
        if (tracer.runmoretrace == 0)
        {
            //tracer.VramOutWriteFile();
            //tracer.GTEoutWriteFile();
            //tracer.MDECOutWriteFile(true);
            //tracer.CDOutWriteFile(true);
            //tracer.PadOutWriteFile(true);
            tracer.SPUOutWriteFile(true);
            //tracer.trace_file_last();
            int a = 5;
        }
    }
    else if (tracer.runmoretrace > 0)
    {
        tracer.totalticks += g_state.lastticks;
        tracer.sumticks += g_state.lastticks;
    }

    tracer.debug_outdivcnt = (tracer.debug_outdivcnt + 1) % tracer.debug_outdiv;

    tracer.commands++;
}

template<PGXPMode pgxp_mode, bool debug>
static void ExecuteImpl()
{
  g_using_interpreter = true;
  g_state.frame_done = false;
  while (!g_state.frame_done)
  {
      g_state.afterCommand = false;

    //TimingEvents::UpdateCPUDowncount();

    //while (g_state.pending_ticks < g_state.downcount)
    {
      if (tracer.commands == 0) CPU::g_state.pending_ticks = 0;
      trace(g_state);
      tracer.isException = false;
      tracer.isInterrupt = false;
      g_gpu->m_pending_command_ticks_last = 0;
      g_mdec.hadTranfer = false;

      //if (tracer.commands == 10) { tracer.overwriteButtons = true; tracer.overwriteByte0 = 0xFF; tracer.overwriteByte1 = 0xBF; } // cross
      //if (tracer.commands == 100000) { tracer.overwriteButtons = true; tracer.overwriteByte0 = 0xFF; tracer.overwriteByte1 = 0xFF; } // cross
      //if (tracer.commands == 1000000) { tracer.overwriteButtons = true; tracer.overwriteByte0 = 0xFE; tracer.overwriteByte1 = 0xFF; } // select

      if (tracer.traclist_ptr == 715962)
      //if (tracer.traclist_ptr == 0x25c5c)
      {
        int xx = 0;

        //std::unique_ptr<ByteStream> stream = FileSystem::OpenFile("C:\\Projekte\\psx\\duckstation\\states\\state.sav", BYTESTREAM_OPEN_CREATE | BYTESTREAM_OPEN_WRITE | BYTESTREAM_OPEN_TRUNCATE | BYTESTREAM_OPEN_ATOMIC_UPDATE | BYTESTREAM_OPEN_STREAMED);
        //System::SaveState(stream.get());
        //stream->Commit();
      }

      if (HasPendingInterrupt() && !g_state.interrupt_delay)
        DispatchInterrupt();

      if (g_state.interrupt_delay > 0) g_state.interrupt_delay--;
      g_state.pending_ticks++;

      // now executing the instruction we previously fetched
      g_state.current_instruction.bits = g_state.next_instruction.bits;
      g_state.current_instruction_pc = g_state.regs.pc;
      g_state.current_instruction_in_branch_delay_slot = g_state.next_instruction_is_branch_delay_slot;
      g_state.current_instruction_was_branch_taken = g_state.branch_was_taken;
      g_state.next_instruction_is_branch_delay_slot = false;
      g_state.branch_was_taken = false;
      g_state.exception_raised = false;

      // fetch the next instruction - even if this fails, it'll still refetch on the flush so we can continue
      if (!FetchInstruction())
      {
          tracer.commands--;
          tracer.totalticks--;
          if (tracer.debug_outdivcnt == 0) tracer.traclist_ptr--;
          else tracer.debug_outdivcnt--;
          g_state.pending_ticks--;
          continue;
      }

      // execute the instruction we previously fetched
      ExecuteInstruction<pgxp_mode, debug>();
      g_state.afterCommand = true;
      // next load delay
      UpdateLoadDelay();
    }

    TimingEvents::RunEvents();
  }
}

void Execute()
{
  if (g_settings.gpu_pgxp_enable)
  {
    if (g_settings.gpu_pgxp_cpu)
      ExecuteImpl<PGXPMode::CPU, false>();
    else
      ExecuteImpl<PGXPMode::Memory, false>();
  }
  else
  {
    ExecuteImpl<PGXPMode::Disabled, false>();
  }
}

void ExecuteDebug()
{
  if (g_settings.gpu_pgxp_enable)
  {
    if (g_settings.gpu_pgxp_cpu)
      ExecuteImpl<PGXPMode::CPU, true>();
    else
      ExecuteImpl<PGXPMode::Memory, true>();
  }
  else
  {
    ExecuteImpl<PGXPMode::Disabled, true>();
  }
}

void SingleStep()
{
  s_single_step = true;
  ExecuteDebug();
  g_host_interface->ReportFormattedDebuggerMessage("Stepped to 0x%08X.", g_state.regs.pc);
}

namespace CodeCache {

template<PGXPMode pgxp_mode>
void InterpretCachedBlock(const CodeBlock& block)
{
  // set up the state so we've already fetched the instruction
  DebugAssert(g_state.regs.pc == block.GetPC());
  g_state.regs.npc = block.GetPC() + 4;

  for (const CodeBlockInstruction& cbi : block.instructions)
  {
    g_state.pending_ticks++;

    // now executing the instruction we previously fetched
    g_state.current_instruction.bits = cbi.instruction.bits;
    g_state.current_instruction_pc = cbi.pc;
    g_state.current_instruction_in_branch_delay_slot = cbi.is_branch_delay_slot;
    g_state.current_instruction_was_branch_taken = g_state.branch_was_taken;
    g_state.branch_was_taken = false;
    g_state.exception_raised = false;

    // update pc
    g_state.regs.pc = g_state.regs.npc;
    g_state.regs.npc += 4;

    // execute the instruction we previously fetched
    ExecuteInstruction<pgxp_mode, false>();

    // next load delay
    UpdateLoadDelay();

    if (g_state.exception_raised)
      break;
  }

  // cleanup so the interpreter can kick in if needed
  g_state.next_instruction_is_branch_delay_slot = false;
}

template void InterpretCachedBlock<PGXPMode::Disabled>(const CodeBlock& block);
template void InterpretCachedBlock<PGXPMode::Memory>(const CodeBlock& block);
template void InterpretCachedBlock<PGXPMode::CPU>(const CodeBlock& block);

template<PGXPMode pgxp_mode>
void InterpretUncachedBlock()
{
  g_state.regs.npc = g_state.regs.pc;
  if (!FetchInstructionForInterpreterFallback())
    return;

  // At this point, pc contains the last address executed (in the previous block). The instruction has not been fetched
  // yet. pc shouldn't be updated until the fetch occurs, that way the exception occurs in the delay slot.
  bool in_branch_delay_slot = false;
  for (;;)
  {
    g_state.pending_ticks++;

    // now executing the instruction we previously fetched
    g_state.current_instruction.bits = g_state.next_instruction.bits;
    g_state.current_instruction_pc = g_state.regs.pc;
    g_state.current_instruction_in_branch_delay_slot = g_state.next_instruction_is_branch_delay_slot;
    g_state.current_instruction_was_branch_taken = g_state.branch_was_taken;
    g_state.next_instruction_is_branch_delay_slot = false;
    g_state.branch_was_taken = false;
    g_state.exception_raised = false;

    // Fetch the next instruction, except if we're in a branch delay slot. The "fetch" is done in the next block.
    const bool branch = IsBranchInstruction(g_state.current_instruction);
    if (!g_state.current_instruction_in_branch_delay_slot || branch)
    {
      if (!FetchInstructionForInterpreterFallback())
        break;
    }
    else
    {
      g_state.regs.pc = g_state.regs.npc;
    }

    // execute the instruction we previously fetched
    ExecuteInstruction<pgxp_mode, false>();

    // next load delay
    UpdateLoadDelay();

    if (g_state.exception_raised || (!branch && in_branch_delay_slot) ||
        IsExitBlockInstruction(g_state.current_instruction))
    {
      break;
    }

    in_branch_delay_slot = branch;
  }
}

template void InterpretUncachedBlock<PGXPMode::Disabled>();
template void InterpretUncachedBlock<PGXPMode::Memory>();
template void InterpretUncachedBlock<PGXPMode::CPU>();

} // namespace CodeCache

namespace Recompiler::Thunks {

bool InterpretInstruction()
{
  ExecuteInstruction<PGXPMode::Disabled, false>();
  return g_state.exception_raised;
}

bool InterpretInstructionPGXP()
{
  ExecuteInstruction<PGXPMode::Memory, false>();
  return g_state.exception_raised;
}

} // namespace Recompiler::Thunks

} // namespace CPU
