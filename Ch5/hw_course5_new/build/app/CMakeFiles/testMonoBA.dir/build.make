# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/build

# Include any dependencies generated for this target.
include app/CMakeFiles/testMonoBA.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include app/CMakeFiles/testMonoBA.dir/compiler_depend.make

# Include the progress variables for this target.
include app/CMakeFiles/testMonoBA.dir/progress.make

# Include the compile flags for this target's objects.
include app/CMakeFiles/testMonoBA.dir/flags.make

app/CMakeFiles/testMonoBA.dir/TestMonoBA.cpp.o: app/CMakeFiles/testMonoBA.dir/flags.make
app/CMakeFiles/testMonoBA.dir/TestMonoBA.cpp.o: ../app/TestMonoBA.cpp
app/CMakeFiles/testMonoBA.dir/TestMonoBA.cpp.o: app/CMakeFiles/testMonoBA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object app/CMakeFiles/testMonoBA.dir/TestMonoBA.cpp.o"
	cd /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT app/CMakeFiles/testMonoBA.dir/TestMonoBA.cpp.o -MF CMakeFiles/testMonoBA.dir/TestMonoBA.cpp.o.d -o CMakeFiles/testMonoBA.dir/TestMonoBA.cpp.o -c /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/app/TestMonoBA.cpp

app/CMakeFiles/testMonoBA.dir/TestMonoBA.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testMonoBA.dir/TestMonoBA.cpp.i"
	cd /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/app/TestMonoBA.cpp > CMakeFiles/testMonoBA.dir/TestMonoBA.cpp.i

app/CMakeFiles/testMonoBA.dir/TestMonoBA.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testMonoBA.dir/TestMonoBA.cpp.s"
	cd /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/app/TestMonoBA.cpp -o CMakeFiles/testMonoBA.dir/TestMonoBA.cpp.s

# Object files for target testMonoBA
testMonoBA_OBJECTS = \
"CMakeFiles/testMonoBA.dir/TestMonoBA.cpp.o"

# External object files for target testMonoBA
testMonoBA_EXTERNAL_OBJECTS =

app/testMonoBA: app/CMakeFiles/testMonoBA.dir/TestMonoBA.cpp.o
app/testMonoBA: app/CMakeFiles/testMonoBA.dir/build.make
app/testMonoBA: backend/libslam_course_backend.a
app/testMonoBA: app/CMakeFiles/testMonoBA.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testMonoBA"
	cd /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/build/app && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testMonoBA.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
app/CMakeFiles/testMonoBA.dir/build: app/testMonoBA
.PHONY : app/CMakeFiles/testMonoBA.dir/build

app/CMakeFiles/testMonoBA.dir/clean:
	cd /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/build/app && $(CMAKE_COMMAND) -P CMakeFiles/testMonoBA.dir/cmake_clean.cmake
.PHONY : app/CMakeFiles/testMonoBA.dir/clean

app/CMakeFiles/testMonoBA.dir/depend:
	cd /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/app /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/build /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/build/app /home/wrk/File/SLAM/My_course/VIO/Ch5/hw_course5_new/build/app/CMakeFiles/testMonoBA.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : app/CMakeFiles/testMonoBA.dir/depend

