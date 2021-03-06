/** @file cmdline.h
 *  @brief The header file for the command line option parser
 *  generated by GNU Gengetopt version 2.22.6
 *  http://www.gnu.org/software/gengetopt.
 *  DO NOT modify this file, since it can be overwritten
 *  @author GNU Gengetopt by Lorenzo Bettini */

#ifndef CMDLINE_H
#define CMDLINE_H

/* If we use autoconf.  */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h> /* for FILE */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef CMDLINE_PARSER_PACKAGE
/** @brief the program name (used for printing errors) */
#define CMDLINE_PARSER_PACKAGE "imagenet_tensorrt_builder"
#endif

#ifndef CMDLINE_PARSER_PACKAGE_NAME
/** @brief the complete program name (used for help and version) */
#define CMDLINE_PARSER_PACKAGE_NAME "imagenet_tensorrt_builder"
#endif

#ifndef CMDLINE_PARSER_VERSION
/** @brief the program version */
#define CMDLINE_PARSER_VERSION "0.1"
#endif

/** @brief Where the command line options are stored */
struct gengetopt_args_info
{
  const char *help_help; /**< @brief Print help and exit help description.  */
  const char *version_help; /**< @brief Print version and exit help description.  */
  char * dir_arg;	/**< @brief path to directory name contains model json and weights exported by dump_chainer.py or dump_caffe.py.  */
  char * dir_orig;	/**< @brief path to directory name contains model json and weights exported by dump_chainer.py or dump_caffe.py original value given at command line.  */
  const char *dir_help; /**< @brief path to directory name contains model json and weights exported by dump_chainer.py or dump_caffe.py help description.  */
  int gpu_arg;	/**< @brief GPU ID (default='0').  */
  char * gpu_orig;	/**< @brief GPU ID original value given at command line.  */
  const char *gpu_help; /**< @brief GPU ID help description.  */
  char * model_arg;	/**< @brief specify where to save the built model.  */
  char * model_orig;	/**< @brief specify where to save the built model original value given at command line.  */
  const char *model_help; /**< @brief specify where to save the built model help description.  */
  char * mode_arg;	/**< @brief specify mode (default='fp32').  */
  char * mode_orig;	/**< @brief specify mode original value given at command line.  */
  const char *mode_help; /**< @brief specify mode help description.  */
  char * calib_arg;	/**< @brief specify filename of calibration image list if using mode=int8 and do calibration from scratch.  */
  char * calib_orig;	/**< @brief specify filename of calibration image list if using mode=int8 and do calibration from scratch original value given at command line.  */
  const char *calib_help; /**< @brief specify filename of calibration image list if using mode=int8 and do calibration from scratch help description.  */
  char * in_cache_arg;	/**< @brief specify filename to calibration cache when using mode=int8.  */
  char * in_cache_orig;	/**< @brief specify filename to calibration cache when using mode=int8 original value given at command line.  */
  const char *in_cache_help; /**< @brief specify filename to calibration cache when using mode=int8 help description.  */
  char * out_cache_arg;	/**< @brief specify output filename to calibration cache if using mode=int8 (--calib also needs to be specified) (default='').  */
  char * out_cache_orig;	/**< @brief specify output filename to calibration cache if using mode=int8 (--calib also needs to be specified) original value given at command line.  */
  const char *out_cache_help; /**< @brief specify output filename to calibration cache if using mode=int8 (--calib also needs to be specified) help description.  */
  int max_batch_arg;	/**< @brief specify the maximum batch size this model is supposed to receive (default='1').  */
  char * max_batch_orig;	/**< @brief specify the maximum batch size this model is supposed to receive original value given at command line.  */
  const char *max_batch_help; /**< @brief specify the maximum batch size this model is supposed to receive help description.  */
  int workspace_arg;	/**< @brief specify workspace size in GB that TensorRT is allowed to use while building the network (default='4').  */
  char * workspace_orig;	/**< @brief specify workspace size in GB that TensorRT is allowed to use while building the network original value given at command line.  */
  const char *workspace_help; /**< @brief specify workspace size in GB that TensorRT is allowed to use while building the network help description.  */
  const char *verbose_help; /**< @brief Verbose mode help description.  */
  
  unsigned int help_given ;	/**< @brief Whether help was given.  */
  unsigned int version_given ;	/**< @brief Whether version was given.  */
  unsigned int dir_given ;	/**< @brief Whether dir was given.  */
  unsigned int gpu_given ;	/**< @brief Whether gpu was given.  */
  unsigned int model_given ;	/**< @brief Whether model was given.  */
  unsigned int mode_given ;	/**< @brief Whether mode was given.  */
  unsigned int calib_given ;	/**< @brief Whether calib was given.  */
  unsigned int in_cache_given ;	/**< @brief Whether in-cache was given.  */
  unsigned int out_cache_given ;	/**< @brief Whether out-cache was given.  */
  unsigned int max_batch_given ;	/**< @brief Whether max-batch was given.  */
  unsigned int workspace_given ;	/**< @brief Whether workspace was given.  */
  unsigned int verbose_given ;	/**< @brief Whether verbose was given.  */

} ;

/** @brief The additional parameters to pass to parser functions */
struct cmdline_parser_params
{
  int override; /**< @brief whether to override possibly already present options (default 0) */
  int initialize; /**< @brief whether to initialize the option structure gengetopt_args_info (default 1) */
  int check_required; /**< @brief whether to check that all required options were provided (default 1) */
  int check_ambiguity; /**< @brief whether to check for options already specified in the option structure gengetopt_args_info (default 0) */
  int print_errors; /**< @brief whether getopt_long should print an error message for a bad option (default 1) */
} ;

/** @brief the purpose string of the program */
extern const char *gengetopt_args_info_purpose;
/** @brief the usage string of the program */
extern const char *gengetopt_args_info_usage;
/** @brief the description string of the program */
extern const char *gengetopt_args_info_description;
/** @brief all the lines making the help output */
extern const char *gengetopt_args_info_help[];

/**
 * The command line parser
 * @param argc the number of command line options
 * @param argv the command line options
 * @param args_info the structure where option information will be stored
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int cmdline_parser (int argc, char **argv,
  struct gengetopt_args_info *args_info);

/**
 * The command line parser (version with additional parameters - deprecated)
 * @param argc the number of command line options
 * @param argv the command line options
 * @param args_info the structure where option information will be stored
 * @param override whether to override possibly already present options
 * @param initialize whether to initialize the option structure my_args_info
 * @param check_required whether to check that all required options were provided
 * @return 0 if everything went fine, NON 0 if an error took place
 * @deprecated use cmdline_parser_ext() instead
 */
int cmdline_parser2 (int argc, char **argv,
  struct gengetopt_args_info *args_info,
  int override, int initialize, int check_required);

/**
 * The command line parser (version with additional parameters)
 * @param argc the number of command line options
 * @param argv the command line options
 * @param args_info the structure where option information will be stored
 * @param params additional parameters for the parser
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int cmdline_parser_ext (int argc, char **argv,
  struct gengetopt_args_info *args_info,
  struct cmdline_parser_params *params);

/**
 * Save the contents of the option struct into an already open FILE stream.
 * @param outfile the stream where to dump options
 * @param args_info the option struct to dump
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int cmdline_parser_dump(FILE *outfile,
  struct gengetopt_args_info *args_info);

/**
 * Save the contents of the option struct into a (text) file.
 * This file can be read by the config file parser (if generated by gengetopt)
 * @param filename the file where to save
 * @param args_info the option struct to save
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int cmdline_parser_file_save(const char *filename,
  struct gengetopt_args_info *args_info);

/**
 * Print the help
 */
void cmdline_parser_print_help(void);
/**
 * Print the version
 */
void cmdline_parser_print_version(void);

/**
 * Initializes all the fields a cmdline_parser_params structure 
 * to their default values
 * @param params the structure to initialize
 */
void cmdline_parser_params_init(struct cmdline_parser_params *params);

/**
 * Allocates dynamically a cmdline_parser_params structure and initializes
 * all its fields to their default values
 * @return the created and initialized cmdline_parser_params structure
 */
struct cmdline_parser_params *cmdline_parser_params_create(void);

/**
 * Initializes the passed gengetopt_args_info structure's fields
 * (also set default values for options that have a default)
 * @param args_info the structure to initialize
 */
void cmdline_parser_init (struct gengetopt_args_info *args_info);
/**
 * Deallocates the string fields of the gengetopt_args_info structure
 * (but does not deallocate the structure itself)
 * @param args_info the structure to deallocate
 */
void cmdline_parser_free (struct gengetopt_args_info *args_info);

/**
 * Checks that all the required options were specified
 * @param args_info the structure to check
 * @param prog_name the name of the program that will be used to print
 *   possible errors
 * @return
 */
int cmdline_parser_required (struct gengetopt_args_info *args_info,
  const char *prog_name);

extern const char *cmdline_parser_mode_values[];  /**< @brief Possible values for mode. */


#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* CMDLINE_H */
