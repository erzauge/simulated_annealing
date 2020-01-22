#include "Logging.hpp"

bool Logger::quiet = false;             ///< Log only to file or also to stdout
int Logger::verbosity = 0;              ///< global verbosity level to use
std::string Logger::logfilename = "";   ///< Filename to write the messenges to

Logger::Logger(log_level_t level, std::string file, int line, std::string function)
    : level(level),
      ss(),
      file(file),
      line(line),
      function(function)
{
    if(level <= verbosity)
    {
        ss.precision(12);

        #ifdef __unix__
        if(isatty(fileno(stdout))) // Terminal -> use colors
            label = CLABEL[level];
        else
        #endif
            label = LABEL[level]; // Windows or file -> do not use color
    }
}

/// Writes the message to the logfile and flushes it to stdout, according to settings.
Logger::~Logger()
{
    if(level <= verbosity)
    {
        // write current thread, if we are multithreaded
        // #ifdef _OPENMP
        // if(omp_get_num_threads() > 1)
        //     ss << " (thread " << omp_get_thread_num() << ")";
        // #endif

        if(level <= LOG_WARNING)
            ss << " (" << file << ":" << line << " [" << function << "()]) ";

        if(!quiet)
            std::cout << label << ss.str() << std::endl;

        if(!logfilename.empty())
        {
            std::ofstream logfile(logfilename, std::ofstream::app);
            logfile << LABEL[level] << ss.str() << "\n";
        }
    }
}