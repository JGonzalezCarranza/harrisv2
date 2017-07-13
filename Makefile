TARGET=harrisdemo

#########################################################################################

# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

# These flags will override any settings
ifeq ($(i386),1)
	OS_SIZE = 32
	mainOS_ARCH = i686
endif

ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif

# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= /usr/local/cuda-8.0
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= g++

# Extra user flags
EXTRA_NVCCFLAGS ?= -O2 -use_fast_math -std=c++11 -lpthread -lX11
EXTRA_LDFLAGS   ?=

# CUDA code generation flags
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
#no sirve para operaciones atomicas
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30,code=compute_30 -gencode arch=compute_32,code=sm_32,code=compute_32 -gencode arch=compute_35,code=sm_35,code=compute_35 -gencode arch=compute_37,code=sm_37,code=compute_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50,code=compute_50 -gencode arch=compute_52,code=sm_52,code=compute_52 -gencode arch=compute_53,code=sm_53,code=compute_53
GENCODE_SM60	:= -gencode arch=compute_60,code=sm_60,code=compute_60
GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30)
#GENCODE_FLAGS   := $(GENCODE_SM20)
# OS-specific build flags
ifneq ($(DARWIN),)
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -framework CUDA -lcudart
      CCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcuda -lcudart
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcuda -lcudart
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

# Debug build flags
#ifeq ($(dbg),1)
      #CCFLAGS   += -g
      #NVCCFLAGS += -g -G
#      TARGET := debug
#else
#    #  TARGET := release
#endif

INCLUDES      := -I$(CUDA_INC_PATH) -I. -I$(HOME)/NVIDIA_CUDA-8.0_Samples/common/inc -I/usr/include/



#########################################################################################

all: $(TARGET)

clean:
	rm -f $(OBJFILES) *.o
	rm -f $(TARGET)


calcMax.o: calcMax.cu
	$(NVCC) $(EXTRA_NVCCFLAGS) $(INCLUDES) -c calcMax.cu

goodPixels.o: goodPixels.cu
	$(NVCC) $(EXTRA_NVCCFLAGS) $(INCLUDES) -c goodPixels.cu

quicksortinverse.o: quicksortinverse.cu
	$(NVCC) $(EXTRA_NVCCFLAGS) $(INCLUDES) -c quicksortinverse.cu

gradiente.o: gradiente.cu
	$(NVCC) $(EXTRA_NVCCFLAGS)  $(INCLUDES) -c gradiente.cu

harrisResponse.o: harrisResponse.cu
	$(NVCC) $(EXTRA_NVCCFLAGS) $(INCLUDES) -c harrisResponse.cu

harrisHilos.o: harrisHilos.cu
	$(NVCC) $(EXTRA_NVCCFLAGS) $(INCLUDES) -c harrisHilos.cu

histograma.o: histograma.cu
	$(NVCC) $(EXTRA_NVCCFLAGS) $(INCLUDES) -c histograma.cu

reserva.o: reserva.cu
	$(NVCC) $(EXTRA_NVCCFLAGS) $(INCLUDES) -c reserva.cu

harrisdemo: goodPixels.o quicksortinverse.o gradiente.o harrisResponse.o histograma.o calcMax.o reserva.o harrisHilos.o
	$(NVCC) $(INCLUDES)  -o harrisdemo calcMax.o goodPixels.o quicksortinverse.o gradiente.o harrisResponse.o histograma.o reserva.o harrisHilos.o -lX11
	rm -f *.o
