include(FetchContent)  

FetchContent_Declare(	
    spdlog	
	GIT_REPOSITORY https://github.com/gabime/spdlog.git
	GIT_TAG v1.9.2 
	SOURCE_DIR ${PROJECT_SOURCE_DIR}/3rdparty_downloads/spdlog 
)

set(SPDLOG_INSTALL ON)
FetchContent_MakeAvailable(spdlog)
