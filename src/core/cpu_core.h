#pragma once
#include "common/bitfield.h"
#include "cpu_types.h"
#include "gte_types.h"
#include "cpu_core.h"
#include "types.h"
#include <array>
#include <optional>
#include <vector>

class StateWrapper;

namespace CPU {

enum : VirtualMemoryAddress
{
  RESET_VECTOR = UINT32_C(0xBFC00000)
};
enum : PhysicalMemoryAddress
{
  DCACHE_LOCATION = UINT32_C(0x1F800000),
  DCACHE_LOCATION_MASK = UINT32_C(0xFFFFFC00),
  DCACHE_OFFSET_MASK = UINT32_C(0x000003FF),
  DCACHE_SIZE = UINT32_C(0x00000400),
  ICACHE_SIZE = UINT32_C(0x00001000),
  ICACHE_SLOTS = ICACHE_SIZE / sizeof(u32),
  ICACHE_LINE_SIZE = 16,
  ICACHE_LINES = ICACHE_SIZE / ICACHE_LINE_SIZE,
  ICACHE_SLOTS_PER_LINE = ICACHE_SLOTS / ICACHE_LINES,
  ICACHE_TAG_ADDRESS_MASK = 0xFFFFFFF0u,
  ICACHE_INVALID_BITS = 0x0Fu,
};

union CacheControl
{
  u32 bits;

  BitField<u32, bool, 0, 1> lock_mode;
  BitField<u32, bool, 1, 1> invalidate_mode;
  BitField<u32, bool, 2, 1> tag_test_mode;
  BitField<u32, bool, 3, 1> dcache_scratchpad;
  BitField<u32, bool, 7, 1> dcache_enable;
  BitField<u32, u8, 8, 2> icache_fill_size; // actually dcache? icache always fills to 16 bytes
  BitField<u32, bool, 11, 1> icache_enable;
};

struct State
{
  // ticks the CPU has executed
  TickCount lastticks = 0;
  TickCount pending_ticks = 0;
  TickCount downcount = 0;

  Registers regs = {};
  Cop0Registers cop0_regs = {};
  Instruction next_instruction = {};

  // address of the instruction currently being executed
  Instruction current_instruction = {};
  u32 current_instruction_pc = 0;
  bool current_instruction_in_branch_delay_slot = false;
  bool current_instruction_was_branch_taken = false;
  bool next_instruction_is_branch_delay_slot = false;
  bool branch_was_taken = false;
  bool exception_raised = false;
  u8 interrupt_delay = false;
  bool frame_done = false;

  // load delays
  Reg load_delay_reg = Reg::count;
  u32 load_delay_value = 0;
  Reg next_load_delay_reg = Reg::count;
  u32 next_load_delay_value = 0;

  CacheControl cache_control{0};

  // GTE registers are stored here so we can access them on ARM with a single instruction
  GTE::Regs gte_regs = {};

  // 4 bytes of padding here on x64
  bool use_debug_dispatcher = false;

  u8* fastmem_base = nullptr;

  bool afterCommand;
  bool slowDMAIRQ;

  // data cache (used as scratchpad)
  std::array<u8, DCACHE_SIZE> dcache = {};
  std::array<u32, ICACHE_LINES> icache_tags = {};
  std::array<u8, ICACHE_SIZE> icache_data = {};

  static constexpr u32 GPRRegisterOffset(u32 index) { return offsetof(State, regs.r) + (sizeof(u32) * index); }
  static constexpr u32 GTERegisterOffset(u32 index) { return offsetof(State, gte_regs.r32) + (sizeof(u32) * index); }
};

extern State g_state;
extern bool g_using_interpreter;

void Initialize();
void Shutdown();
void Reset();
bool DoState(StateWrapper& sw);
void ClearICache();
void UpdateFastmemBase();

/// Executes interpreter loop.
void Execute();
void ExecuteDebug();
void SingleStep();

// Forces an early exit from the CPU dispatcher.
void ForceDispatcherExit();

ALWAYS_INLINE Registers& GetRegs()
{
  return g_state.regs;
}

ALWAYS_INLINE TickCount GetPendingTicks()
{
  return g_state.pending_ticks;
}
ALWAYS_INLINE void ResetPendingTicks()
{
  g_state.lastticks = g_state.pending_ticks;
  g_state.pending_ticks = 0;
}
ALWAYS_INLINE void AddPendingTicks(TickCount ticks)
{
  g_state.pending_ticks += ticks;
}

// state helpers
ALWAYS_INLINE bool InUserMode()
{
  return g_state.cop0_regs.sr.KUc;
}
ALWAYS_INLINE bool InKernelMode()
{
  return !g_state.cop0_regs.sr.KUc;
}

// Memory reads variants which do not raise exceptions.
bool SafeReadMemoryByte(VirtualMemoryAddress addr, u8* value);
bool SafeReadMemoryHalfWord(VirtualMemoryAddress addr, u16* value);
bool SafeReadMemoryWord(VirtualMemoryAddress addr, u32* value);
bool SafeWriteMemoryByte(VirtualMemoryAddress addr, u8 value);
bool SafeWriteMemoryHalfWord(VirtualMemoryAddress addr, u16 value);
bool SafeWriteMemoryWord(VirtualMemoryAddress addr, u32 value);

// External IRQs
void SetExternalInterrupt(u8 bit);
void ClearExternalInterrupt(u8 bit);

void DisassembleAndPrint(u32 addr);
void DisassembleAndLog(u32 addr);
void DisassembleAndPrint(u32 addr, u32 instructions_before, u32 instructions_after);

// Write to CPU execution log file.
void WriteToExecutionLog(const char* format, ...) printflike(1, 2);

// Trace Routines
bool IsTraceEnabled();
void StartTrace();
void StopTrace();

// Breakpoint callback - if the callback returns false, the breakpoint will be removed.
using BreakpointCallback = bool (*)(VirtualMemoryAddress address);

struct Breakpoint
{
  VirtualMemoryAddress address;
  BreakpointCallback callback;
  u32 number;
  u32 hit_count;
  bool auto_clear;
  bool enabled;
};

using BreakpointList = std::vector<Breakpoint>;

// Breakpoints
bool HasAnyBreakpoints();
bool HasBreakpointAtAddress(VirtualMemoryAddress address);
BreakpointList GetBreakpointList(bool include_auto_clear = false, bool include_callbacks = false);
bool AddBreakpoint(VirtualMemoryAddress address, bool auto_clear = false, bool enabled = true);
bool AddBreakpointWithCallback(VirtualMemoryAddress address, BreakpointCallback callback);
bool RemoveBreakpoint(VirtualMemoryAddress address);
void ClearBreakpoints();
bool AddStepOverBreakpoint();
bool AddStepOutBreakpoint(u32 max_instructions_to_search = 1000);

extern bool TRACE_EXECUTION;

// ############################## export

class cpustate
{
public:
  uint32_t ticks;
  uint32_t newticks;
  uint32_t regs[32];
  uint32_t regs_hi;
  uint32_t regs_lo;
  uint32_t sr;  
  uint32_t cause;  
  uint32_t pc;  
  uint32_t opcode; 

  uint16_t irq;

  uint16_t gpu_time;
  uint16_t gpu_line;
  uint32_t gpu_stat;
  uint16_t fifocount;
  uint16_t gpu_ticks;

  uint32_t mdec_stat;

  uint16_t cd_status;

  uint16_t timer[3];

  unsigned char debug8;
  uint16_t debug16;
  uint32_t debug32;

  void update(State g_state);
};

class Tracer
{
public:
  uint32_t totalticks = 0;
  uint32_t sumticks = 0;

  int startindex;

  int additional_steps;
  uint32_t commands;
  uint32_t cyclenr;
  bool tracenext;

  const int Tracelist_Length = 2000000;
  cpustate Tracelist[2000000];
  int debug_outdivcnt = 0;
  int debug_outdiv = 1;

  int traclist_ptr;
  int runmoretrace = -1;

  bool isException;
  bool isInterrupt;
  uint32_t exceptionOpcode;

  uint32_t nextTimingEventDelay;

  bool overwriteButtons = false;
  u8 overwriteByte0 = 0;
  u8 overwriteByte1 = 0;
  bool forceanalog = false;

  void trace_file_last();

//#define VRAMFILEOUT
//#define VRAMPIXELOUT
#ifdef VRAMFILEOUT
  uint32_t debug_VramOutTime[1000000];
  uint32_t debug_VramOutAddr[1000000];
  uint16_t debug_VramOutData[1000000];
  uint8_t debug_VramOutType[1000000];
  uint32_t debug_VramOutCount;
#endif
  void VramOutWriteFile();

//#define GTEFILEOUT
#ifdef GTEFILEOUT
  uint32_t debug_GTEOutTime[1000000];
  uint8_t debug_GTEOutAddr[1000000];
  uint32_t debug_GTEOutData[1000000];
  uint8_t debug_GTEOutType[1000000];
  uint32_t debug_GTEOutCount;
  uint32_t debug_GTELast[64];
#endif
  void GTEoutRegCapture(uint8_t regtype);
  void GTEoutCommandCapture(uint32_t command);
  void GTEoutWriteFile();
  void GTETest();

//#define MDECFILEOUT
#ifdef MDECFILEOUT
  uint32_t debug_MDECOutTime[1000000];
  uint8_t  debug_MDECOutAddr[1000000];
  uint32_t debug_MDECOutData[1000000];
  uint8_t  debug_MDECOutType[1000000];
  uint32_t debug_MDECOutCount;
#endif
  void MDECOutCapture(uint8_t type, uint8_t addr, uint32_t data);
  void MDECOutWriteFile(bool writeTest);
  void MDECTest();

//#define CDFILEOUT
#ifdef CDFILEOUT
#define CDFILEOUTMAX 1000000
  uint32_t debug_CDOutTime[CDFILEOUTMAX];
  uint8_t  debug_CDOutAddr[CDFILEOUTMAX];
  uint32_t debug_CDOutData[CDFILEOUTMAX];
  uint8_t  debug_CDOutType[CDFILEOUTMAX];
  uint32_t debug_CDOutCount;
  uint8_t  debug_XAOutType[1000000];
  uint32_t debug_XAOutData[1000000];
  uint32_t debug_XAOutCount;
#endif
  void CDOutCapture(uint8_t type, uint8_t addr, uint32_t data);
  void XAOutCapture(uint8_t type, uint32_t data);
  void CDOutWriteFile(bool writeTest);
  void CDTest();

//#define PADFILEOUT
#ifdef PADFILEOUT
  uint32_t debug_PadOutTime[1000000];
  uint8_t  debug_PadOutAddr[1000000];
  uint16_t debug_PadOutData[1000000];
  uint8_t  debug_PadOutType[1000000];
  uint32_t debug_PadOutCount;
#endif
  void PadOutCapture(uint8_t addr, uint16_t data, uint8_t type);
  void PadOutWriteFile(bool writeTest);
  void PadTest();

#define SPUFILEOUT
#ifdef SPUFILEOUT
#define SPUFILEOUTMAX 5000000
  uint32_t debug_SPUOutTime[SPUFILEOUTMAX];
  uint16_t  debug_SPUOutAddr[SPUFILEOUTMAX];
  uint16_t debug_SPUOutData[SPUFILEOUTMAX];
  uint8_t  debug_SPUOutType[SPUFILEOUTMAX];
  uint32_t debug_SPUOutCount;
  bool debug_SPUOutAll = false;
#endif
  void SPUOutCapture(uint16_t addr, uint16_t data, uint8_t type, uint8_t timeadd);
  void SPUOutWriteFile(bool writeTest);
  void SPUTest();

};
extern Tracer tracer;

} // namespace CPU
