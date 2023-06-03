/**
 * @file   Msg.h
 * @author Yibo Lin
 * @date   Jan 2019
 */

#ifndef DREAMPLACE_UTILITY_MSG_H
#define DREAMPLACE_UTILITY_MSG_H

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include "utility/src/namespace.h"

DREAMPLACE_BEGIN_NAMESPACE

/// message type for print functions
enum MessageType {
  kNONE = 0,
  kINFO = 1,
  kWARN = 2,
  kERROR = 3,
  kDEBUG = 4,
  kASSERT = 5
};

/// print to screen (stdout)
int dreamplacePrint(MessageType m, const char* format, ...);
/// print to stream
int dreamplacePrintStream(MessageType m, FILE* stream, const char* format, ...);
/// core function to print formatted data from variable argument list
int dreamplaceVPrintStream(MessageType m, FILE* stream, const char* format,
                           va_list args);
/// format to a buffer
int dreamplaceSPrint(MessageType m, char* buf, const char* format, ...);
/// core function to format a buffer
int dreamplaceVSPrint(MessageType m, char* buf, const char* format,
                      va_list args);
/// format prefix
int dreamplaceSPrintPrefix(MessageType m, char* buf);

/// assertion
void dreamplacePrintAssertMsg(const char* expr, const char* fileName,
                              unsigned lineNum, const char* funcName,
                              const char* format, ...);
void dreamplacePrintAssertMsg(const char* expr, const char* fileName,
                              unsigned lineNum, const char* funcName);

#define dreamplaceAssertMsg(condition, args...)                       \
  do {                                                                \
    if (!(condition)) {                                               \
      ::DREAMPLACE_NAMESPACE::dreamplacePrintAssertMsg(               \
          #condition, __FILE__, __LINE__, __PRETTY_FUNCTION__, args); \
      abort();                                                        \
    }                                                                 \
  } while (false)
#define dreamplaceAssert(condition)                             \
  do {                                                          \
    if (!(condition)) {                                         \
      ::DREAMPLACE_NAMESPACE::dreamplacePrintAssertMsg(         \
          #condition, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
      abort();                                                  \
    }                                                           \
  } while (false)

/// static assertion
template <bool>
struct dreamplaceStaticAssert;
template <>
struct dreamplaceStaticAssert<true> {
  dreamplaceStaticAssert(const char* = NULL) {}
};

DREAMPLACE_END_NAMESPACE

#endif
