#pragma once

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#ifndef MISC3D_DISABLE_LOGGER

#ifndef __FUNC_NAME__
#ifdef WIN32  // WINDOWS
#define __FUNC_NAME__ __FUNCTION__
#else  //*NIX
#define __FUNC_NAME__ __func__
#endif
#endif

namespace misc3d {

enum LoggerLevel {
    LoggerLevel_Trace = spdlog::level::trace,
    LoggerLevel_Debug = spdlog::level::debug,
    LoggerLevel_Info = spdlog::level::info,
    LoggerLevel_Warn = spdlog::level::warn,
    LoggerLevel_Error = spdlog::level::err,
    LoggerLevel_Critical = spdlog::level::critical,
    LoggerLevel_Off = spdlog::level::off,
};

static bool GetLoggerFilePath(char* name) {
    if (name == nullptr)
        return false;
#ifdef WIN32
    LPITEMIDLIST pidl{NULL};

    SHGetSpecialFolderLocation(NULL, CSIDL_LOCAL_APPDATA, &pidl);
    if (pidl) {
        SHGetPathFromIDListA(pidl, name);
        CoTaskMemFree(pidl);
    }
    if (strlen(name) > 120) {
        return false;
    }

    sprintf(name, "%slog/misc3d.log", name);
    return true;
#else
    strcpy(name, "log/misc3d.log");
    return true;
#endif
}

static std::shared_ptr<spdlog::logger> GetOrCreateLogger(const char* logname) {
    auto lp = spdlog::get(logname);
    if (!lp) {
        std::vector<spdlog::sink_ptr> sinks;
        char filePath[256];
        GetLoggerFilePath(filePath);
        auto fp =
            std::make_shared<spdlog::sinks::rotating_file_sink_mt>(filePath, 1024 * 1024 * 10, 0);
        sinks.push_back(fp);
#ifdef ENABLE_LOGGER_FILE
        auto stdp = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        sinks.push_back(stdp);
#endif
        lp = std::make_shared<spdlog::logger>(logname, begin(sinks), end(sinks));
        lp->flush_on(spdlog::level::debug);
        spdlog::register_logger(lp);
    }
    return lp;
}

static auto sg_misc3d_logger = GetOrCreateLogger("Misc3D");

#define MISC3D_SET_LEVEL(level) \
    sg_misc3d_logger->set_level(static_cast<spdlog::level::level_enum>(level))

#define MISC3D_INFO(...)                                                             \
    sg_misc3d_logger->info("{0}:{1} : {2} : {3}", __FILE__, __LINE__, __FUNC_NAME__, \
                           fmt::format(__VA_ARGS__))
#define MISC3D_DEBUG(...)                                                             \
    sg_misc3d_logger->debug("{0}:{1} : {2} : {3}", __FILE__, __LINE__, __FUNC_NAME__, \
                            fmt::format(__VA_ARGS__))
#define MISC3D_WARN(...)                                                             \
    sg_misc3d_logger->warn("{0}:{1} : {2} : {3}", __FILE__, __LINE__, __FUNC_NAME__, \
                           fmt::format(__VA_ARGS__))
#define MISC3D_ERROR(...)                                                             \
    sg_misc3d_logger->error("{0}:{1} : {2} : {3}", __FILE__, __LINE__, __FUNC_NAME__, \
                            fmt::format(__VA_ARGS__))

#define MISC3D_ENABLE_BT(n) sg_misc3d_logger->enable_backtrace(n)
#define MISC3D_DUMP_BT() sg_misc3d_logger->dump_backtrace()
#define MISC3D_DISABLE_BT() sg_misc3d_logger->disable_backtrace()

}  // namespace misc3d

#endif