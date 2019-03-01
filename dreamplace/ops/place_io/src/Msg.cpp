/*************************************************************************
    > File Name: Msg.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Fri 31 Jul 2015 03:20:14 PM CDT
 ************************************************************************/

#include "Msg.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

GPF_BEGIN_NAMESPACE

int gpfPrint(MessageType m, const char* format, ...)
{
	va_list args;
	va_start(args, format);
	int ret = gpfVPrintStream(m, stdout, format, args);
	va_end(args);

	return ret;
}

int gpfPrintStream(MessageType m, FILE* stream, const char* format, ...)
{
	va_list args;
	va_start(args, format);
	int ret = gpfVPrintStream(m, stream, format, args);
	va_end(args);

	return ret;
}

int gpfVPrintStream(MessageType m, FILE* stream, const char* format, va_list args)
{
	// print prefix 
    char prefix[8];
    gpfSPrintPrefix(m, prefix);
	fprintf(stream, "%s", prefix);

	// print message 
	int ret = vfprintf(stream, format, args);
	
	return ret;
}

int gpfSPrint(MessageType m, char* buf, const char* format, ...)
{
	va_list args;
	va_start(args, format);
	int ret = gpfVSPrint(m, buf, format, args);
	va_end(args);

	return ret;
}

int gpfVSPrint(MessageType m, char* buf, const char* format, va_list args)
{
	// print prefix 
    char prefix[8];
    gpfSPrintPrefix(m, prefix);
	sprintf(buf, "%s", prefix);

	// print message 
	int ret = vsprintf(buf+strlen(prefix), format, args);
	
	return ret;
}

int gpfSPrintPrefix(MessageType m, char* prefix)
{
	switch (m)
	{
		case kNONE:
            return sprintf(prefix, "%c", '\0');
		case kINFO:
			return sprintf(prefix, "(I) ");
		case kWARN:
            return sprintf(prefix, "(W) ");
		case kERROR:
            return sprintf(prefix, "(E) ");
		case kDEBUG:
            return sprintf(prefix, "(D) ");
        case kASSERT:
            return sprintf(prefix, "(A) ");
		default:
			gpfAssertMsg(0, "unknown message type");
	}
    return 0;
}

void gpfPrintAssertMsg(const char* expr, const char* fileName, unsigned lineNum, const char* funcName, const char* format, ...)
{
    // construct message 
    char buf[1024];
    va_list args;
	va_start(args, format);
    vsprintf(buf, format, args);
    va_end(args);

    // print message 
    gpfPrintStream(kASSERT, stderr, "%s:%u: %s: Assertion `%s' failed: %s\n", fileName, lineNum, funcName, expr, buf);
}

void gpfPrintAssertMsg(const char* expr, const char* fileName, unsigned lineNum, const char* funcName)
{
    // print message
    gpfPrintStream(kASSERT, stderr, "%s:%u: %s: Assertion `%s' failed\n", fileName, lineNum, funcName, expr);
}

GPF_END_NAMESPACE
