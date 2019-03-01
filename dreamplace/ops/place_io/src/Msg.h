/*************************************************************************
    > File Name: Msg.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Fri 31 Jul 2015 03:18:29 PM CDT
 ************************************************************************/

#ifndef GPF_MSG_H
#define GPF_MSG_H

#include <cstdarg>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include "Namespace.h"

GPF_BEGIN_NAMESPACE

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
int gpfPrint(MessageType m, const char* format, ...);
/// print to stream 
int gpfPrintStream(MessageType m, FILE* stream, const char* format, ...);
/// core function to print formatted data from variable argument list 
int gpfVPrintStream(MessageType m, FILE* stream, const char* format, va_list args);
/// format to a buffer 
int gpfSPrint(MessageType m, char* buf, const char* format, ...);
/// core function to format a buffer 
int gpfVSPrint(MessageType m, char* buf, const char* format, va_list args);
/// format prefix 
int gpfSPrintPrefix(MessageType m, char* buf);

/// assertion 
void gpfPrintAssertMsg(const char* expr, const char* fileName, unsigned lineNum, const char* funcName, const char* format, ...);
void gpfPrintAssertMsg(const char* expr, const char* fileName, unsigned lineNum, const char* funcName);

#define gpfAssertMsg(condition, args...) do {\
    if (!(condition)) \
    {\
        ::GPF_NAMESPACE::gpfPrintAssertMsg(#condition, __FILE__, __LINE__, __PRETTY_FUNCTION__, args); \
        abort(); \
    }\
} while (false)
#define gpfAssert(condition) do {\
    if (!(condition)) \
    {\
        ::GPF_NAMESPACE::gpfPrintAssertMsg(#condition, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
        abort(); \
    }\
} while (false)

/// static assertion 
template <bool>
struct gpfStaticAssert;
template <>
struct gpfStaticAssert<true> 
{
    gpfStaticAssert(const char* = NULL) {}
};


GPF_END_NAMESPACE

#endif
