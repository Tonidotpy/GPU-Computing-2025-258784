#include "logger.h"

#include <string.h>

/*! \brief Terminal color codes */
#define LOGGER_COLOR_CODE_BLACK "30"
#define LOGGER_COLOR_CODE_RED "31"
#define LOGGER_COLOR_CODE_GREEN "32"
#define LOGGER_COLOR_CODE_YELLOW "33"
#define LOGGER_COLOR_CODE_BLUE "34"
#define LOGGER_COLOR_CODE_MAGENTA "35"
#define LOGGER_COLOR_CODE_CYAN "36"
#define LOGGER_COLOR_CODE_WHITE "37"

/*!
 * \brief String prefix to add to the message for each log level
 */
const char *logger_prefix[] = {
    [LOGGER_LEVEL_ERROR] = "error: ",
    [LOGGER_LEVEL_WARNING] = "warning: ",
    [LOGGER_LEVEL_INFO] = "info: ",
    [LOGGER_LEVEL_DEBUG] = "debug: "
};

/*!
 * \brief Color codes for each log level
 */
const char *logger_color_code[] = {
    [LOGGER_LEVEL_ERROR] = LOGGER_COLOR_CODE_RED,
    [LOGGER_LEVEL_WARNING] = LOGGER_COLOR_CODE_YELLOW,
    [LOGGER_LEVEL_INFO] = LOGGER_COLOR_CODE_BLUE,
    [LOGGER_LEVEL_DEBUG] = LOGGER_COLOR_CODE_GREEN
};

/*!
 * \brief Base function to log a formatted string to a file given its log level
 *      and the format arguments
 *
 * \param hlogger A pointer to the logger handler
 * \param log_level The message log level
 * \param fp The pointer to the file where the log will be saved
 * \param fmt String that will be formatted and printed
 * \params args List of arguments needed to format the string
 */
void _logger_vfprintf(const char *file, int line, const LoggerHandler_t *hlogger, LoggerLevel_t log_level, FILE *fp, const char *fmt, va_list args) {
    if (hlogger == NULL || fp == NULL || fmt == NULL)
        return;
    if (log_level >= LOGGER_LEVEL_COUNT || log_level == LOGGER_LEVEL_NONE)
        return;

    // Log only if the message log level is lower or equal to the logger log level
    if (log_level <= hlogger->level) {
        if (hlogger->colors_enable)
            fprintf(fp, "\033[%sm", logger_color_code[log_level]);
        fprintf(fp, "%s: %d: %s", file, line, logger_prefix[log_level]);
        if (hlogger->colors_enable)
            fprintf(fp, "\033[0m");
        vfprintf(fp, fmt, args);
    }
}

void logger_init(LoggerHandler_t *hlogger, LoggerLevel_t log_level, bool colors_enable) {
    if (hlogger == NULL)
        return;
    memset(hlogger, 0U, sizeof(*hlogger));
    hlogger->level = log_level;
    hlogger->colors_enable = colors_enable;
}

void logger_set_level(LoggerHandler_t *hlogger, LoggerLevel_t log_level) {
    if (hlogger == NULL)
        return;
    hlogger->level = log_level;
}

bool logger_are_colors_enabled(LoggerHandler_t *hlogger) {
    if (hlogger == NULL)
        return false;
    return hlogger->colors_enable;
}

void logger_set_colors_enable(LoggerHandler_t *hlogger, bool colors_enable) {
    if (hlogger == NULL)
        return;
    hlogger->colors_enable = colors_enable;
}

void _logger_printf(const char *file, int line, const LoggerHandler_t *hlogger, LoggerLevel_t log_level, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    _logger_vfprintf(file, line, hlogger, log_level, stdout, fmt, args);
    va_end(args);
}

void _logger_fprintf(const char *file, int line, const LoggerHandler_t *hlogger, LoggerLevel_t log_level, FILE *fp, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    _logger_vfprintf(file, line, hlogger, log_level, fp, fmt, args);
    va_end(args);
}

void _logger_error(const char *file, int line, const LoggerHandler_t *hlogger, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    _logger_vfprintf(file, line, hlogger, LOGGER_LEVEL_ERROR, stderr, fmt, args);
    va_end(args);
}

void _logger_warning(const char *file, int line, const LoggerHandler_t *hlogger, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    _logger_vfprintf(file, line, hlogger, LOGGER_LEVEL_WARNING, stderr, fmt, args);
    va_end(args);
}

void _logger_info(const char *file, int line, const LoggerHandler_t *hlogger, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    _logger_vfprintf(file, line, hlogger, LOGGER_LEVEL_INFO, stdout, fmt, args);
    va_end(args);
}

void _logger_debug(const char *file, int line, const LoggerHandler_t *hlogger, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    _logger_vfprintf(file, line, hlogger, LOGGER_LEVEL_DEBUG, stdout, fmt, args);
    va_end(args);
}
