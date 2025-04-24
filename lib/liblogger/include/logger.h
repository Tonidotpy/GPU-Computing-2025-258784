#ifndef LOGGER_H
#define LOGGER_H

#include <stdarg.h>
#include <stdio.h>
#include <stdbool.h>

/*!
 * @brief Wrapper macros to call the logger functions
 */
#ifdef NLOGGER

#define logger_printf(hlogger, log_level, fmt, ...) \
    do {                                            \
        UNUSED(hlogger);                            \
        UNUSED(log_level);                          \
        UNUSED(fmt);                                \
    } while (0)
#define logger_fprintf(hlogger, log_level, fp, fmt, ...) \
    do {                                                 \
        UNUSED(hlogger);                                 \
        UNUSED(log_level);                               \
        UNUSED(fp);                                      \
        UNUSED(fmt);                                     \
    } while (0)
#define logger_error(hlogger, fmt, ...) \
    do {                                \
        UNUSED(hlogger);                \
        UNUSED(fmt);                    \
    } while (0)
#define logger_warning(hlogger, fmt, ...) \
    do {                                  \
        UNUSED(hlogger);                  \
        UNUSED(fmt);                      \
    } while (0)
#define logger_info(hlogger, fmt, ...) \
    do {                               \
        UNUSED(hlogger);               \
        UNUSED(fmt);                   \
    } while (0)
#define logger_debug(hlogger, fmt, ...) \
    do {                                \
        UNUSED(hlogger);                \
        UNUSED(fmt);                    \
    } while (0)

#else // NLOGGER

#define logger_printf(hlogger, log_level, fmt, ...) _logger_printf(__FILE__, __LINE__, hlogger, log_level, fmt, __VA_ARGS__)
#define logger_fprintf(hlogger, log_level, fp, fmt, ...) _logger_fprintf(__FILE__, __LINE__, hlogger, log_level, fp, fmt, __VA_ARGS__)
#define logger_error(hlogger, fmt, ...) _logger_error(__FILE__, __LINE__, hlogger, fmt, __VA_ARGS__)
#define logger_warning(hlogger, fmt, ...) _logger_warning(__FILE__, __LINE__, hlogger, fmt, __VA_ARGS__)
#define logger_info(hlogger, fmt, ...) _logger_info(__FILE__, __LINE__, hlogger, fmt, __VA_ARGS__)
#define logger_debug(hlogger, fmt, ...) _logger_debug(__FILE__, __LINE__, hlogger, fmt, __VA_ARGS__)

#endif // NLOGGER

/*!
 * \brief Definition of the possible log levels
 *
 * \details For every level all the messages belonging to the current and previous
 *      levels are shown (e.g. info will show info, warning and error messages)
 */
typedef enum _LoggerLevel_t {
    LOGGER_LEVEL_NONE = 0,
    LOGGER_LEVEL_ERROR,
    LOGGER_LEVEL_WARNING,
    LOGGER_LEVEL_INFO,
    LOGGER_LEVEL_DEBUG,
    LOGGER_LEVEL_COUNT
} LoggerLevel_t;

/*!
 * \brief Logger handler structure definition
 *
 * \var level The current log level
 * \var colors_enable Flag to enable or disable colors
 */
typedef struct _LoggerHandler_t {
    LoggerLevel_t level;
    bool colors_enable;
} LoggerHandler_t;

/*!
 * \brief Initialize the logger handler structure
 *
 * \param hlogger A pointer to the logger handler
 * \param log_level The log level of the logger
 * \param enable_colors Enable or disable logger colors
 */
void logger_init(LoggerHandler_t *hlogger, LoggerLevel_t log_level, bool colors_enable);

/*!
 * \brief Change the log level of the logger
 *
 * \param hlogger A pointer to the logger handler
 * \param log_level The new log level to set
 */
void logger_set_level(LoggerHandler_t *hlogger, LoggerLevel_t log_level);

/*!
 * \brief Check if the colors are enabled
 *
 * \param hlogger A pointer to the logger handler
 *
 * \return bool True if the colors are enabled, false otherwise
 */
bool logger_are_colors_enabled(LoggerHandler_t *hlogger);

/*!
 * \brief Enable or disable logger colors
 *
 * \param hlogger A pointer to the logger handler
 * \param colors_enable True to enable colors, false to disable them
 */
void logger_set_colors_enable(LoggerHandler_t *hlogger, bool colors_enable);

/*!
 * \brief Log to standard output
 *
 * \param file The name of the file where the log is printed
 * \param line The line number of the file where the log is printed
 * \param hlogger A pointer to the logger handler
 * \param log_level The message log level
 * \param fmt String that will be formatted and printed
 * \params ... Additional parameters needed to format the string
 */
void _logger_printf(const char *file, int line, const LoggerHandler_t *hlogger, LoggerLevel_t log_level, const char *fmt, ...);

/*!
 * \brief Log to the given file
 *
 * \details The given file can also be stdout and stderr
 *
 * \param file The name of the file where the log is printed
 * \param line The line number of the file where the log is printed
 * \param hlogger A pointer to the logger handler
 * \param log_level The message log level
 * \param fp The pointer to the file where the log will be saved
 * \param fmt String that will be formatted and printed
 * \params ... Additional parameters needed to format the string
 */
void _logger_fprintf(const char *file, int line, const LoggerHandler_t *hlogger, LoggerLevel_t log_level, FILE *fp, const char *fmt, ...);

/*!
 * \brief Log an error message to standard error
 *
 * \param file The name of the file where the log is printed
 * \param line The line number of the file where the log is printed
 * \param hlogger A pointer to the logger handler
 * \param fmt String that will be formatted and printed
 * \params ... Additional parameters needed to format the string
 */
void _logger_error(const char *file, int line, const LoggerHandler_t *hlogger, const char *fmt, ...);

/*!
 * \brief Log a warning message to standard error
 *
 * \param file The name of the file where the log is printed
 * \param line The line number of the file where the log is printed
 * \param hlogger A pointer to the logger handler
 * \param fmt String that will be formatted and printed
 * \params ... Additional parameters needed to format the string
 */
void _logger_warning(const char *file, int line, const LoggerHandler_t *hlogger, const char *fmt, ...);

/*!
 * \brief Log an info message to standard output
 *
 * \param file The name of the file where the log is printed
 * \param line The line number of the file where the log is printed
 * \param hlogger A pointer to the logger handler
 * \param fmt String that will be formatted and printed
 * \params ... Additional parameters needed to format the string
 */
void _logger_info(const char *file, int line, const LoggerHandler_t *hlogger, const char *fmt, ...);

/*!
 * \brief Log a debug message to standard output
 *
 * \param file The name of the file where the log is printed
 * \param line The line number of the file where the log is printed
 * \param hlogger A pointer to the logger handler
 * \param fmt String that will be formatted and printed
 * \params ... Additional parameters needed to format the string
 */
void _logger_debug(const char *file, int line, const LoggerHandler_t *hlogger, const char *fmt, ...);

#endif // LOGGER_H
