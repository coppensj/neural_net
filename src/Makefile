CXX := g++

override INCLUDES = \
	-I.

override CXXFLAGS += \
	$(INCLUDES) \
	-O3 \
	-std=c++17 \
	-Wall \
	-Wextra \
	-fpermissive 

override LDFLAGS += \
	-lstdc++fs \
	-lz \
	-lm

SOURCES := $(wildcard *.cpp) 
OBJS := $(patsubst %.cpp,%.o,$(SOURCES))
DEPS := $(patsubst %.cpp,%.d,$(SOURCES))

.PHONY: all clean

all: test_network

clean:
	@echo "Cleaning..."
	$(RM) $(OBJS) $(DEPS) test_network

# for executable
test_network: $(OBJS) 
	$(CXX) $(CXXFLAGS) $^ -o $@

-include $(DEPS)

# generic rule
%.o: %.cpp Makefile
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@
