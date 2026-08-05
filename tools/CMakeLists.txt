cmake_minimum_required(VERSION 3.14 FATAL_ERROR)


set(EXAMPLES construct_npy query_npy cereal_tests)
foreach(EXAMPLE IN LISTS EXAMPLES)
  add_executable(${EXAMPLE} ${EXAMPLE}.cpp ${HEADERS})
  target_link_libraries(${EXAMPLE} FLAT_NAV_LIB ${CNPY_LIB} ${ZLIB_LIB_RELEASE})

  # This ensures that the executables are placed in the build directory
  set_target_properties(${EXAMPLE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                              "${CMAKE_BINARY_DIR}")
  install(TARGETS ${EXAMPLE} RUNTIME DESTINATION bin)
endforeach(EXAMPLE IN LISTS EXAMPLES)

find_package(Threads REQUIRED)

# distance_server calls libnuma functions unconditionally, so it always needs
# libnuma (install libnuma-dev on Debian/Ubuntu, numactl-devel on RHEL/Fedora).
find_library(NUMA_LIBRARY numa REQUIRED)

add_executable(distance_server distance_server.cpp ${HEADERS})
target_link_libraries(distance_server FLAT_NAV_LIB ${NUMA_LIBRARY} Threads::Threads rt)
set_target_properties(distance_server PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                            "${CMAKE_BINARY_DIR}")
install(TARGETS distance_server RUNTIME DESTINATION bin)

# hbw_search only uses libnuma optionally, behind -DFLATNAV_USE_NUMA (see its
# NumaThreadPool/NumaAllocation usage), so it doesn't link libnuma by default.
add_executable(hbw_search hbw_search.cpp ${HEADERS})
target_compile_definitions(hbw_search PRIVATE FLATNAV_CXL_OFFLOAD)
target_link_libraries(hbw_search FLAT_NAV_LIB Threads::Threads)
set_target_properties(hbw_search PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                            "${CMAKE_BINARY_DIR}")
install(TARGETS hbw_search RUNTIME DESTINATION bin)
