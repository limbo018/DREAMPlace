##
# @file   Include.mk
# @author Yibo Lin
# @date   Mar 2019
#

# ==========================================================================
#                        Compile-time Configurations
# ==========================================================================

# default DBG is off
DBG = 0

ifeq ($(DBG), 1)
	CXXFLAGS = $(CXXFLAGS_DEBUG) -DDEBUG
else
	CXXFLAGS = $(CXXFLAGS_RELEASE)
endif

SITE_PACKAGE = # --user or empty 


LIMBO_ROOT_DIR = $(PROJECT_ROOT_DIR)/thirdparty/Limbo

# ==========================================================================
#                                Compilers
# ==========================================================================

MAKE = make
NVCC = nvcc 
AR = ar
include $(LIMBO_ROOT_DIR)/limbo/makeutils/FindCompiler.mk

# ==========================================================================
#                                Compilation Flags
# ==========================================================================

ifneq ($(findstring clang, $(CXX)), ) # CXX contains clang 
	CXXFLAGS_BASIC = -ferror-limit=1 -fPIC -W -Wall -Wextra -Wreturn-type -m64 -std=c++11 -Wno-deprecated -Wno-unused-parameter -Wno-unused-local-typedef
	CXXFLAGS_DEBUG = -g -DDEBUG $(CXXFLAGS_BASIC) 
	CXXFLAGS_RELEASE = -O3 $(CXXFLAGS_BASIC) 

	CFLAGS_BASIC = -ferror-limit=1 -fPIC -W -Wall -Wextra -Wreturn-type -m64 -Wno-deprecated -Wno-unused-parameter -Wno-unused-local-typedef
	CFLAGS_DEBUG = -g -DDEBUG $(CFLAGS_BASIC) 
	CFLAGS_RELEASE = -O3 $(CFLAGS_BASIC) 

	ARFLAGS = rvs
else 
	CXXFLAGS_BASIC = -fmax-errors=1 -fPIC -W -Wall -Wextra -Wreturn-type -m64 -std=c++11 -Wno-deprecated -Wno-unused-local-typedefs -Wno-ignored-qualifiers
	CXXFLAGS_DEBUG = -g $(CXXFLAGS_BASIC) 
	CXXFLAGS_RELEASE = -O3 -fopenmp $(CXXFLAGS_BASIC) 

	CFLAGS_BASIC = -fmax-errors=1 -fPIC -W -Wall -Wextra -Wreturn-type -ansi -m64 -Wno-deprecated -Wno-unused-local-typedefs
	CFLAGS_DEBUG = -g $(CFLAGS_BASIC) 
	CFLAGS_RELEASE = -O3 -fopenmp $(CFLAGS_BASIC) 

	ARFLAGS = rvs
endif 

# dependency to Boost and get BOOST_LINK_FLAG
ifdef BOOST_DIR
include $(LIMBO_ROOT_DIR)/limbo/makeutils/FindBoost.mk
endif
# dependency to Zlib and get ZLIB_LINK_FLAG
ifdef ZLIB_DIR
include $(LIMBO_ROOT_DIR)/limbo/makeutils/FindZlib.mk
endif

# ==========================================================================
#                                 Doxygen
# ==========================================================================

DOXYGEN = doxygen
