CC=g++
INCLUDE_DIRS=include/
CXXFLAGS=-I$(INCLUDE_DIRS)
BUILD_DIR=build/

default: driver

network.o:
	$(CC) $(CXXFLAGS) -c src/cpp-nn/network.cpp -o $(BUILD_DIR)$@

Agent.o:
	$(CC) $(CXXFLAGS) -c src/dqn/Agent.cpp -o $(BUILD_DIR)$@

utils.o: 
	$(CC) $(CXXFLAGS) -c src/envtools/utils.cpp -o $(BUILD_DIR)$@

driver: network.o Agent.o utils.o
	$(CC) $(CXXFLAGS) src/driver.cpp $(BUILD_DIR)network.o $(BUILD_DIR)Agent.o $(BUILD_DIR)utils.o -o bin/$@

driver_nn: network.o Agent.o utils.o
	$(CC) $(CXXFLAGS) src/driver_nn.cpp $(BUILD_DIR)network.o $(BUILD_DIR)Agent.o $(BUILD_DIR)utils.o -o bin/$@

clean:
	rm -rf build/*.o bin/driver