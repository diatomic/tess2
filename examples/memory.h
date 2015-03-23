#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

// From http://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-run-time-in-c

//////////////////////////////////////////////////////////////////////////////
//
// process_mem_usage(double &, double &) - takes two doubles by reference,
// attempts to read the system-dependent data for a process' virtual memory
// size and resident set size, and return the results in KB.
//
// On failure, returns 0.0, 0.0

inline void process_mem_usage(size_t& vm_usage, size_t& resident_set)
{
   using std::ios_base;
   using std::ifstream;
   using std::string;

   vm_usage     = 0;
   resident_set = 0;

#ifdef __linux
   // 'file' stat seems to give the most reliable results
   //
   ifstream stat_stream("/proc/self/stat",ios_base::in);

   // dummy vars for leading entries in stat that we don't care about
   //
   string pid, comm, state, ppid, pgrp, session, tty_nr;
   string tpgid, flags, minflt, cminflt, majflt, cmajflt;
   string utime, stime, cutime, cstime, priority, nice;
   string O, itrealvalue, starttime;

   // the two fields we want
   //
   unsigned long vsize;
   long rss;

   stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
               >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
               >> utime >> stime >> cutime >> cstime >> priority >> nice
               >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

   stat_stream.close();

   long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
   vm_usage     = vsize / 1024;
   resident_set = rss * page_size_kb;
#endif
}

inline size_t process_mem_usage()
{
   size_t vm, rss;
   process_mem_usage(vm, rss);
   return rss;
}

inline size_t proc_status_value(const std::string& field)
{
#ifdef __linux
    std::ifstream in("/proc/self/status");
    std::string line;
    while (in)
    {
        std::getline(in, line);
        if (line.compare(0, field.length(), field) == 0)
        {
            std::istringstream iss(line);
            std::string f; size_t res;
            iss >> f >> res;
            return res;
        }
    }
#endif
    return 0;
}
