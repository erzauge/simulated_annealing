/*! \file */
#ifndef LOGGING_H
#define LOGGING_H

#ifdef __unix__
#include <cstdio>
#include <unistd.h>
#endif

#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#ifdef _OPENMP
   #include <omp.h>
#endif

#ifdef NLOG
// if compiled with -DNLOG, the compiler hopefully optimizes everything after the else away
#define LOG(level) \
    if(true) {} \
    else Logger(level)
#else
/** Low overhead logging macro.
 *
 * Prevents construction of the Logger object and evaluation of << operators
 * if the Logger::verbosity level is lower than the \a level of the message.
 * This also enables the output of Filename, Line and Function for every output
 *
 * Inspired by: \n
 *   http://stackoverflow.com/a/11826787/1698412 \n
 *   http://stackoverflow.com/questions/1255576/what-is-good-practice-for-generating-verbose-output
 */
#define LOG(level) \
    if(level > Logger::verbosity) {} \
    else Logger(level, __FILE__, __LINE__, __func__)
#endif

/// Enum defining which verbosity level to use, i.e. which events to output
enum log_level_t {
    LOG_QUIET = 0,  ///< Do not log anything
    LOG_ALWAYS,     ///< Do log things that should always be logged
    LOG_ERROR,      ///< Do log critical error messages which terminate the process
    LOG_WARNING,    ///< Do log warnings (recommended level)
    LOG_INFO,       ///< Do log interesting informations
    LOG_TIMING,     ///< Do log measured timings
    LOG_DEBUG,      ///< Do log debug messages
    LOG_ALL,        ///< Do log all messages
    LOG_TOO_MUCH    ///< Do log everything (this might result probably in log files measured in GB)
};

/// Labels appearing in front of messenges of the associated verbosity levels
static const std::vector<std::string> LABEL = {
    "",
    "",
    "Error:   ",
    "Warning: ",
    "Info:    ",
    "Timing:  ",
    "Debug:   ",
    "Debug2:  ",
    "Debug3:  "
};

/// Labels appearing in front of messenges of the associated verbosity levels with color
static const std::vector<std::string> CLABEL = {
    "",
    "",
    "\x1B[31mError:   \033[0m",
    "\x1B[33mWarning: \033[0m",
    "\x1B[34mInfo:    \033[0m",
    "\x1B[34mTiming:  \033[0m",
    "\x1B[34mDebug:   \033[0m",
    "\x1B[34mDebug2:  \033[0m",
    "\x1B[34mDebug3:  \033[0m"
};

/** Logs messages depending on a runtime set verbosity level.
 *
 *  Use it through the #LOG(level) macro.
 */
class Logger {
    public:
        Logger(log_level_t level, std::string file="", int line=0, std::string function="");

        ~Logger();

        static bool quiet;
        static int verbosity;
        static std::string logfilename;

        template<class T>
        friend std::ostream& operator<<(Logger &&l, const T &obj);


    protected:
        log_level_t level;
        std::stringstream ss;

        std::string label;

        std::string file;
        int line;
        std::string function;
};

template<class T>
std::ostream& operator<<(Logger &&l, const T &obj)
{
    l.ss << " " << obj;
    return l.ss;
}

template<class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T> &v)
{
    for(auto i : v)
        os << i << " ";
    return os;
}

#endif