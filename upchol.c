/*
 * Benchmark program for uplooking Cholesky factorisation
 *
 * Copyright (C) 2023 James D. Trotter
 *
 * This program is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Benchmark program for uplooking Cholesky factorisation.
 *
 * Authors:
 *
 *  James D. Trotter <james@simula.no>
 *
 */

#include <errno.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <zlib.h>

#include <unistd.h>

#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char * program_name = "upchol";
const char * program_version = "1.0";
const char * program_copyright =
    "Copyright (C) 2023 James D. Trotter";
const char * program_license =
    "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law.";
const char * program_invocation_name;
const char * program_invocation_short_name;

/**
 * ‘program_options’ contains data to related program options.
 */
struct program_options
{
    char * Apath;
    int gzip;
    double fillfactor;
    double growthfactor;
    bool fillcount;
    bool symbolic;
    int verify;
    int progress_interval;
    int verbose;
    int quiet;
    int help;
    int version;
};

/**
 * ‘program_options_init()’ configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->Apath = NULL;
    args->gzip = 0;
    args->fillfactor = 15.0;
    args->growthfactor = 2.0;
    args->fillcount = false;
    args->symbolic = false;
    args->verify = 0;
    args->quiet = 0;
    args->progress_interval = 0;
    args->verbose = 0;
    args->help = 0;
    args->version = 0;
    return 0;
}

/**
 * ‘program_options_free()’ frees memory and other resources
 * associated with parsing program options.
 */
static void program_options_free(
    struct program_options * args)
{
    if (args->Apath) free(args->Apath);
}

/**
 * ‘program_options_print_usage()’ prints a usage text.
 */
static void program_options_print_usage(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] A\n", program_name);
}

/**
 * ‘program_options_print_help()’ prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Perform Cholesky factorisation of symmetric, positive definite,\n");
    fprintf(f, " sparse matrices.\n");
    fprintf(f, "\n");
    fprintf(f, " A symmetric, positive definite matrix ‘A’ is decomposed into\n");
    fprintf(f, " a product ‘A=LL'’ of a lower triangular matrix ‘L’, called\n");
    fprintf(f, " the Cholesky factor, and its (upper triangular) transpose ‘L'’.\n");
    fprintf(f, "\n");
    fprintf(f, " Positional arguments are:\n");
    fprintf(f, "  A                     path to Matrix Market file for the matrix A\n");
    fprintf(f, "\n");
    fprintf(f, " options for factorisation are:\n");
    fprintf(f, "  --fill-factor NUM     estimated ratio of nonzeros in L to A. [15.0]\n");
    fprintf(f, "  --growth-factor NUM   factor used to increase storage for L\n");
    fprintf(f, "                        when estimated fill is exceeded. [2.0]\n");
    fprintf(f, "  --fill-count          compute the amount of fill-in and exit\n");
    fprintf(f, "  --symbolic            perform symbolic factorisation only\n");
    fprintf(f, "  --verify              compute the error ‖A-LL'‖\n");
    fprintf(f, "  --progress N          print progress every N seconds. [0]\n");
    fprintf(f, "\n");
    fprintf(f, " Other options are:\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip    filter files through gzip\n");
    fprintf(f, "  -q, --quiet           do not print Matrix Market output\n");
    fprintf(f, "  -v, --verbose         be more verbose\n");
    fprintf(f, "\n");
    fprintf(f, "  -h, --help            display this help and exit\n");
    fprintf(f, "  --version             display version information and exit\n");
    fprintf(f, "\n");
    fprintf(f, "Report bugs to: <james@simula.no>\n");
}

/**
 * ‘program_options_print_version()’ prints version information.
 */
static void program_options_print_version(
    FILE * f)
{
    fprintf(f, "%s %s\n", program_name, program_version);
    fprintf(f, "%s\n", program_copyright);
    fprintf(f, "%s\n", program_license);
}

/**
 * ‘parse_long_long_int()’ parses a string to produce a number that
 * may be represented with the type ‘long long int’.
 */
static int parse_long_long_int(
    const char * s,
    char ** outendptr,
    int base,
    long long int * out_number,
    int64_t * bytes_read)
{
    errno = 0;
    char * endptr;
    long long int number = strtoll(s, &endptr, base);
    if ((errno == ERANGE && (number == LLONG_MAX || number == LLONG_MIN)) ||
        (errno != 0 && number == 0))
        return errno;
    if (outendptr) *outendptr = endptr;
    if (bytes_read) *bytes_read += endptr - s;
    *out_number = number;
    return 0;
}

/**
 * ‘parse_int()’ parses a string to produce a number that may be
 * represented as an integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a signed integer, ‘ERANGE’ is returned.
 */
int parse_int(
    int * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(s, endptr, 10, &y, bytes_read);
    if (err) return err;
    if (y < INT_MIN || y > INT_MAX) return ERANGE;
    *x = y;
    return 0;
}

/**
 * ‘parse_int64_t()’ parses a string to produce a number that may be
 * represented as a signed, 64-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a signed integer, ‘ERANGE’ is returned.
 */
int parse_int64_t(
    int64_t * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(s, endptr, 10, &y, bytes_read);
    if (err) return err;
    if (y < INT64_MIN || y > INT64_MAX) return ERANGE;
    *x = y;
    return 0;
}

/**
 * ‘parse_double()’ parses a string to produce a number that may be
 * represented as ‘double’.
 *
 * The number is parsed using ‘strtod()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘number’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a double, ‘ERANGE’ is returned.
 */
int parse_double(
    double * x,
    const char * s,
    char ** outendptr,
    int64_t * bytes_read)
{
    errno = 0;
    char * endptr;
    *x = strtod(s, &endptr);
    if ((errno == ERANGE && (*x == HUGE_VAL || *x == -HUGE_VAL)) ||
        (errno != 0 && x == 0)) { return errno; }
    if (outendptr) *outendptr = endptr;
    if (bytes_read) *bytes_read += endptr - s;
    return 0;
}

/**
 * ‘parse_program_options()’ parses program options.
 */
static int parse_program_options(
    int argc,
    char ** argv,
    struct program_options * args,
    int * nargs,
    int * nposargs)
{
    *nargs = 0;
    *nposargs = 0;
    (*nargs)++; argv++;

    /* Parse program options. */
    while (*nargs < argc) {
        if (strcmp(argv[0], "--fill-factor") == 0) {
            if (argc - *nargs < 2) return EINVAL;
            (*nargs)++; argv++;
            int err = parse_double(&args->fillfactor, argv[0], NULL, NULL);
            if (err) return err;
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--fill-factor=") == argv[0]) {
            int err = parse_double(
                &args->fillfactor, argv[0] + strlen("--fill-factor="), NULL, NULL);
            if (err) return err;
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--growth-factor") == 0) {
            if (argc - *nargs < 2) return EINVAL;
            (*nargs)++; argv++;
            int err = parse_double(&args->growthfactor, argv[0], NULL, NULL);
            if (err) return err;
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--growth-factor=") == argv[0]) {
            int err = parse_double(
                &args->growthfactor, argv[0] + strlen("--growth-factor="), NULL, NULL);
            if (err) return err;
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--fill-count") == 0) {
            args->fillcount = true;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--symbolic") == 0) {
            args->symbolic = true;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--verify") == 0) {
            args->verify = true;
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--progress") == 0) {
            if (argc - *nargs < 2) return EINVAL;
            (*nargs)++; argv++;
            int err = parse_int(&args->progress_interval, argv[0], NULL, NULL);
            if (err) return err;
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--progress=") == argv[0]) {
            int err = parse_int(
                &args->progress_interval, argv[0] + strlen("--progress="), NULL, NULL);
            if (err) return err;
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "-z") == 0 ||
            strcmp(argv[0], "--gzip") == 0 ||
            strcmp(argv[0], "--gunzip") == 0 ||
            strcmp(argv[0], "--ungzip") == 0)
        {
            args->gzip = 1;
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "-q") == 0 || strcmp(argv[0], "--quiet") == 0) {
            args->quiet = 1;
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "-v") == 0 || strcmp(argv[0], "--verbose") == 0) {
            args->verbose++;
            (*nargs)++; argv++; continue;
        }

        /* If requested, print program help text. */
        if (strcmp(argv[0], "-h") == 0 || strcmp(argv[0], "--help") == 0) {
            args->help = true;
            (*nargs)++; argv++;
            return 0;
        }

        /* If requested, print program version information. */
        if (strcmp(argv[0], "--version") == 0) {
            args->version = true;
            (*nargs)++; argv++;
            return 0;
        }

        /* Stop parsing options after '--'.  */
        if (strcmp(argv[0], "--") == 0) {
            (*nargs)++; argv++;
            break;
        }

        /*
         * parse positional arguments
         */
        if (*nposargs == 0) {
            args->Apath = strdup(argv[0]);
            if (!args->Apath) return errno;
        } else { return EINVAL; }
        (*nposargs)++;
        (*nargs)++; argv++;
    }
    return 0;
}

/**
 * `timespec_duration()` is the duration, in seconds, elapsed between
 * two given time points.
 */
static double timespec_duration(
    struct timespec t0,
    struct timespec t1)
{
    return (t1.tv_sec - t0.tv_sec) +
        (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/**
 * ‘freadline()’ reads a single line from a stream.
 */
static int freadline(char * linebuf, size_t line_max, int gzip, FILE * f, gzFile gzf) {
    char * s;
    if (!gzip) {
        s = fgets(linebuf, line_max+1, f);
        if (!s && feof(f)) return -1;
        else if (!s) return errno;
    } else {
        s = gzgets(gzf, linebuf, line_max+1);
        if (!s && gzeof(gzf)) return -1;
        else if (!s) return errno;
    }
    int n = strlen(s);
    if (n > 0 && n == line_max && s[n-1] != '\n') return EOVERFLOW;
    return 0;
}

enum mtxobject
{
    mtxmatrix,
    mtxvector,
};

enum mtxformat
{
    mtxarray,
    mtxcoordinate,
};

enum mtxsymmetry
{
    mtxgeneral,
    mtxsymmetric,
};

static int mtxfile_fread_header(
    enum mtxobject * object,
    enum mtxformat * format,
    enum mtxsymmetry * symmetry,
    int64_t * num_rows,
    int64_t * num_columns,
    int64_t * num_nonzeros,
    int gzip,
    FILE * f,
    gzFile gzf,
    int64_t * lines_read,
    int64_t * bytes_read)
{
    int line_max = sysconf(_SC_LINE_MAX);
    char * linebuf = malloc(line_max+1);
    if (!linebuf) return errno;

    /* read and parse header line */
    int err = freadline(linebuf, line_max, gzip, f, gzf);
    if (err) { free(linebuf); return err; }
    char * s = linebuf;
    char * t = s;
    if (strncmp("%%MatrixMarket ", t, strlen("%%MatrixMarket ")) == 0) {
        t += strlen("%%MatrixMarket ");
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;
    if (strncmp("matrix ", t, strlen("matrix ")) == 0) {
        t += strlen("matrix ");
        *object = mtxmatrix;
    } else if (strncmp("vector ", t, strlen("vector ")) == 0) {
        t += strlen("vector ");
        *object = mtxvector;
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;
    if (strncmp("array ", t, strlen("array ")) == 0) {
        t += strlen("array ");
        *format = mtxarray;
    } else if (strncmp("coordinate ", t, strlen("coordinate ")) == 0) {
        t += strlen("coordinate ");
        *format = mtxcoordinate;
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;
    if (strncmp("real ", t, strlen("real ")) == 0) {
        t += strlen("real ");
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;
    if (strncmp("general", t, strlen("general")) == 0) {
        t += strlen("general");
        *symmetry = mtxgeneral;
    } else if (strncmp("symmetric", t, strlen("symmetric")) == 0) {
        t += strlen("symmetric");
        *symmetry = mtxsymmetric;
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;

    /* skip lines starting with '%' */
    do {
        if (lines_read) (*lines_read)++;
        err = freadline(linebuf, line_max, gzip, f, gzf);
        if (err) { free(linebuf); return err; }
        s = t = linebuf;
    } while (linebuf[0] == '%');

    /* parse size line */
    if (*object == mtxmatrix && *format == mtxcoordinate) {
        err = parse_int64_t(num_rows, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
        if (bytes_read) (*bytes_read)++;
        s = t+1;
        err = parse_int64_t(num_columns, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
        if (bytes_read) (*bytes_read)++;
        s = t+1;
        err = parse_int64_t(num_nonzeros, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t) { free(linebuf); return EINVAL; }
        if (lines_read) (*lines_read)++;
    } else if (*object == mtxvector && *format == mtxarray) {
        err = parse_int64_t(num_rows, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t) { free(linebuf); return EINVAL; }
        if (lines_read) (*lines_read)++;
    } else { free(linebuf); return EINVAL; }
    free(linebuf);
    return 0;
}

static int mtxfile_fread_matrix_coordinate_real(
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t * rowidx,
    int64_t * colidx,
    double * a,
    int gzip,
    FILE * f,
    gzFile gzf,
    int64_t * lines_read,
    int64_t * bytes_read)
{
    int line_max = sysconf(_SC_LINE_MAX);
    char * linebuf = malloc(line_max+1);
    if (!linebuf) return errno;
    for (int64_t i = 0; i < num_nonzeros; i++) {
        int err = freadline(linebuf, line_max, gzip, f, gzf);
        if (err) { free(linebuf); return err; }
        char * s = linebuf;
        char * t = s;
        err = parse_int64_t(&rowidx[i], s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
        if (bytes_read) (*bytes_read)++;
        s = t+1;
        err = parse_int64_t(&colidx[i], s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
        if (bytes_read) (*bytes_read)++;
        s = t+1;
        err = parse_double(&a[i], s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t) { free(linebuf); return EINVAL; }
        if (lines_read) (*lines_read)++;
    }
    free(linebuf);
    return 0;
}

static int csr_from_coo_size(
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const double * a,
    int64_t * rowptr,
    int64_t * csrsize,
    int64_t * rowsizemin,
    int64_t * rowsizemax,
    int64_t * diagsize,
    bool symmetric,
    bool lower,
    bool separate_diagonal)
{
#ifdef _OPENMP
    #pragma omp for
#endif
    for (int64_t i = 0; i < num_rows; i++) rowptr[i] = 0;
    rowptr[num_rows] = 0;
    if (num_rows == num_columns && separate_diagonal) {
        if (!symmetric) {
            for (int64_t k = 0; k < num_nonzeros; k++) {
                if (rowidx[k] != colidx[k]) rowptr[rowidx[k]]++;
            }
        } else if (lower) {
            for (int64_t k = 0; k < num_nonzeros; k++) {
                if (rowidx[k] > colidx[k]) rowptr[rowidx[k]]++;
                else if (rowidx[k] < colidx[k]) rowptr[colidx[k]]++;
            }
        } else {
            for (int64_t k = 0; k < num_nonzeros; k++) {
                if (rowidx[k] < colidx[k]) rowptr[rowidx[k]]++;
                else if (rowidx[k] > colidx[k]) rowptr[colidx[k]]++;
            }
        }
    } else {
        if (num_rows != num_columns || !symmetric) {
            for (int64_t k = 0; k < num_nonzeros; k++) rowptr[rowidx[k]]++;
        } else if (lower) {
            for (int64_t k = 0; k < num_nonzeros; k++) {
                if (rowidx[k] >= colidx[k]) rowptr[rowidx[k]]++;
                else if (rowidx[k] < colidx[k]) rowptr[colidx[k]]++;
            }
        } else {
            for (int64_t k = 0; k < num_nonzeros; k++) {
                if (rowidx[k] <= colidx[k]) rowptr[rowidx[k]]++;
                else if (rowidx[k] > colidx[k]) rowptr[colidx[k]]++;
            }
        }
    }
    int64_t rowmin = num_rows > 0 ? rowptr[1] : 0;
    int64_t rowmax = 0;
    for (int64_t i = 1; i <= num_rows; i++) {
        rowmin = rowmin <= rowptr[i] ? rowmin : rowptr[i];
        rowmax = rowmax >= rowptr[i] ? rowmax : rowptr[i];
        rowptr[i] += rowptr[i-1];
    }
    if (num_rows == num_columns && separate_diagonal) { rowmin++; rowmax++; }
    *rowsizemin = rowmin;
    *rowsizemax = rowmax;
    *csrsize = rowptr[num_rows];
    *diagsize = (num_rows == num_columns && separate_diagonal) ? num_rows : 0;
    return 0;
}

static int csr_from_coo(
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const double * a,
    int64_t * rowptr,
    int64_t csrsize,
    int64_t rowsizemin,
    int64_t rowsizemax,
    int64_t * csrcolidx,
    double * csra,
    double * csrad,
    bool symmetric,
    bool lower,
    bool separate_diagonal)
{
    if (num_rows == num_columns && separate_diagonal) {
        if (!symmetric) {
            for (int64_t k = 0; k < num_nonzeros; k++) {
                if (rowidx[k] == colidx[k]) {
                    csrad[rowidx[k]-1] += a[k];
                } else {
                    int64_t i = rowidx[k]-1;
                    csrcolidx[rowptr[i]] = colidx[k]-1;
                    csra[rowptr[i]] = a[k];
                    rowptr[i]++;
                }
            }
        } else if (lower) {
            for (int64_t k = 0; k < num_nonzeros; k++) {
                if (rowidx[k] == colidx[k]) {
                    csrad[rowidx[k]-1] += a[k];
                } else if (rowidx[k] > colidx[k]) {
                    int64_t i = rowidx[k]-1;
                    csrcolidx[rowptr[i]] = colidx[k]-1;
                    csra[rowptr[i]] = a[k];
                    rowptr[i]++;
                } else {
                    int64_t j = colidx[k]-1;
                    csrcolidx[rowptr[j]] = rowidx[k]-1;
                    csra[rowptr[j]] = a[k];
                    rowptr[j]++;
                }
            }
        } else {
            for (int64_t k = 0; k < num_nonzeros; k++) {
                if (rowidx[k] == colidx[k]) {
                    csrad[rowidx[k]-1] += a[k];
                } else if (rowidx[k] < colidx[k]) {
                    int64_t i = rowidx[k]-1;
                    csrcolidx[rowptr[i]] = colidx[k]-1;
                    csra[rowptr[i]] = a[k];
                    rowptr[i]++;
                } else {
                    int64_t j = colidx[k]-1;
                    csrcolidx[rowptr[j]] = rowidx[k]-1;
                    csra[rowptr[j]] = a[k];
                    rowptr[j]++;
                }
            }
        }
    } else {
        if (num_rows != num_columns || !symmetric) {
            for (int64_t k = 0; k < num_nonzeros; k++) {
                int64_t i = rowidx[k]-1;
                csrcolidx[rowptr[i]] = colidx[k]-1;
                csra[rowptr[i]] = a[k];
                rowptr[i]++;
            }
        } else if (lower) {
            for (int64_t k = 0; k < num_nonzeros; k++) {
                if (rowidx[k] >= colidx[k]) {
                    int64_t i = rowidx[k]-1;
                    csrcolidx[rowptr[i]] = colidx[k]-1;
                    csra[rowptr[i]] = a[k];
                    rowptr[i]++;
                } else {
                    int64_t j = colidx[k]-1;
                    csrcolidx[rowptr[j]] = rowidx[k]-1;
                    csra[rowptr[j]] = a[k];
                    rowptr[j]++;
                }
            }
        } else {
            for (int64_t k = 0; k < num_nonzeros; k++) {
                if (rowidx[k] <= colidx[k]) {
                    int64_t i = rowidx[k]-1;
                    csrcolidx[rowptr[i]] = colidx[k]-1;
                    csra[rowptr[i]] = a[k];
                    rowptr[i]++;
                } else {
                    int64_t j = colidx[k]-1;
                    csrcolidx[rowptr[j]] = rowidx[k]-1;
                    csra[rowptr[j]] = a[k];
                    rowptr[j]++;
                }
            }
        }
    }
    for (int64_t i = num_rows; i > 0; i--) rowptr[i] = rowptr[i-1];
    rowptr[0] = 0;

    /* if needed, sort the nonzeros in every row */
#ifdef _OPENMP
    #pragma omp for
#endif
    for (int64_t i = 0; i < num_rows; i++) {
        bool sorted = true;
        for (int64_t k = rowptr[i]+1; k < rowptr[i+1]; k++) {
            if (csrcolidx[k-1] >= csrcolidx[k]) { sorted = false; break; }
        }
        if (sorted) continue;
        fprintf(stderr, "%s: sorting row %'"PRId64"\n", program_invocation_short_name, i+1);
        for (int64_t k = rowptr[i]+1; k < rowptr[i+1]; k++) {
            int64_t j = csrcolidx[k];
            double x = csra[k];
            int64_t l = k-1;
            while (l >= rowptr[i] && csrcolidx[l] > j) {
                csrcolidx[l+1] = csrcolidx[l];
                csra[l+1] = csra[l];
                l--;
            }
            csrcolidx[l+1] = j;
            csra[l+1] = x;
        }
    }
    return 0;
}

/**
 * ‘etree()’ computes an elimination tree for a given symmetric,
 * sparse matrix.
 *
 * The sparsity pattern of the N-by-N matrix A must be provided with
 * its lower triangular entries in compressed sparse row (CSR) format.
 * Thus, ‘Arowptr’ and ‘Acolidx’ are arrays containing row pointers
 * and column offsets, respectively. Moreover, ‘Arowptr’ is an array
 * of length ‘N+1’, whereas ‘Acolidx’ is an array of length
 * ‘Arowptr[N]’.
 *
 * If successful, the parent of each node in the elimination tree is
 * stored in the ‘parent’ array. Moreover, ‘childptr[i]’ points to
 * location of the first child of the ‘i’th node in the array ‘child’.
 * The final entry, ‘childptr[N]’ points one place beyond the last
 * child node of the final node in the elimination tree. In other
 * words, the arrays ‘parent’ and ‘child’ must be of length ‘N’,
 * whereas ‘childptr’ must be of length ‘N+1’.
 */
static int etree(
    int64_t N,
    int64_t * parent,
    int64_t * childptr,
    int64_t * child,
    const int64_t * Arowptr,
    const int64_t * Acolidx)
{
    /*
     * compute parent node in the elimination tree:
     *
     * for each a_ki != 0, where k>i, it suffices to ensure that i is
     * a descendant of k in the next tree T_k.
     */
    for (int64_t k = 0; k < N; k++) {
        parent[k] = -1;
        for (int64_t l = Arowptr[k]; l < Arowptr[k+1]; l++) {
            int64_t i = Acolidx[l];
            int64_t t = i;
            while (parent[t] >= 0 && parent[t] < k) { t = parent[t]; }
            parent[t] = k;
        }
    }

    /* reverse edges to obtain children of each node */
    for (int64_t k = 0; k <= N; k++) childptr[k] = 0;
    for (int64_t k = 0; k < N; k++) { if (parent[k] >= 0) childptr[parent[k]+1]++; }
    for (int64_t k = 1; k <= N; k++) childptr[k] += childptr[k-1];
    for (int64_t k = 0; k < N; k++) {
        if (parent[k] < 0) continue;
        child[childptr[parent[k]]] = k;
        childptr[parent[k]]++;
    }
    for (int64_t k = N; k > 0; k--) childptr[k] = childptr[k-1];
    childptr[0] = 0;
    return 0;
}

/**
 * ‘postorder()’ computes a post-ordering of a given tree.
 *
 * The tree consists of N nodes whose children are given by the
 * ‘child’ array. The children of the ‘i’th node are located at
 * offsets ‘childptr[i]’, ‘childptr[i]+1’, ... ‘childptr[i+1]-1’.
 *
 * If successful, a permutation which places the nodes of the tree in
 * post-order is stored in the array ‘perm’, which must be of length
 * ‘N’.
 */
static int postorder(
    int64_t N,
    int64_t * perm,
    int64_t * level,
    const int64_t * parent,
    const int64_t * childptr,
    const int64_t * child)
{
    int64_t * stack = malloc(N * sizeof(int64_t));
    if (!stack) return errno;

    for (int64_t k = 0, n = 0; k < N; k++) {
        if (parent[k] >= 0) continue;

        /* found a root node in the elimination tree (which may
         * actually be a forest) */
        int64_t prev = -1, p = 0;
        stack[0] = k;
        int64_t stacksize = 1;
        while (stacksize > 0) {
            int64_t v = stack[stacksize-1];
            if (childptr[v] == childptr[v+1]) {
                /* this is a leaf node; pop it off the stack */
                level[n] = p;
                perm[v] = n++;
                stacksize--;
                prev = v;
            } else if (prev >= 0 && v == parent[prev]) {
                /* the node's children have already been visited; pop
                 * the node off the stack and decrease the level. */
                level[n] = --p;
                perm[v] = n++;
                stacksize--;
                prev = v;
            } else {
                /* push children onto the stack in reverse order */
                for (int64_t l = childptr[v+1]-1; l >= childptr[v]; l--)
                    stack[stacksize++] = child[l];
                p++;
            }
        }
    }
    free(stack);
    return 0;
}

/**
 * ‘permute_etree()’ reorders the elimination tree based on a given
 * permutation of its nodes.
 */
static int permute_etree(
    int64_t N,
    int64_t * parent,
    int64_t * childptr,
    int64_t * child,
    const int64_t * perm,
    const int64_t * origparent)
{
    for (int64_t k = 0; k < N; k++) {
        if (origparent[k] >= 0) parent[perm[k]] = perm[origparent[k]];
        else parent[perm[k]] = -1;
    }
    for (int64_t k = 0; k <= N; k++) childptr[k] = 0;
    for (int64_t k = 0; k < N; k++) { if (parent[k] >= 0) childptr[parent[k]+1]++; }
    for (int64_t k = 1; k <= N; k++) childptr[k] += childptr[k-1];
    for (int64_t k = 0; k < N; k++) {
        if (parent[k] < 0) continue;
        child[childptr[parent[k]]] = k;
        childptr[parent[k]]++;
    }
    for (int64_t k = N; k > 0; k--) childptr[k] = childptr[k-1];
    childptr[0] = 0;

    /* if needed, sort the edges of every vertex */
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i = 0; i < N; i++) {
        bool sorted = true;
        for (int64_t k = childptr[i]+1; k < childptr[i+1]; k++) {
            if (child[k-1] >= child[k]) { sorted = false; break; }
        }
        if (sorted) continue;
        for (int64_t k = childptr[i]+1; k < childptr[i+1]; k++) {
            int64_t j = child[k];
            int64_t l = k-1;
            while (l >= childptr[i] && child[l] > j) {
                child[l+1] = child[l];
                l--;
            }
            child[l+1] = j;
        }
    }
    return 0;
}

/**
 * ‘permute()’ reorders a graph given a permutation of its nodes.
 *
 * The original graph is provided in the form of a sparse matrix in
 * compressed sparse row format. This means that a graph of ‘N’
 * vertices is represented by an array ‘origrowptr’ of length ‘N+1’
 * and another array ‘origcolidx’ of length ‘rowptr[N]’.
 *
 * The permuted graph is similarly stored as a sparse matrix in
 * compressed sparse row format using the array ‘rowptr’, which must
 * be of length ‘N+1’, and the array ‘colidx’, which must be of length
 * ‘origrowptr[N]’.
 */
static int permute(
    int64_t N,
    int64_t * rowptr,
    int64_t * colidx,
    const int64_t * perm,
    const int64_t * origrowptr,
    const int64_t * origcolidx)
{
    rowptr[0] = 0;
    for (int64_t i = 0; i < N; i++) rowptr[perm[i]+1] = origrowptr[i+1]-origrowptr[i];
    for (int64_t i = 1; i <= N; i++) rowptr[i] += rowptr[i-1];
    for (int64_t i = 0; i < N; i++) {
        for (int64_t k = origrowptr[i]; k < origrowptr[i+1]; k++) {
            int64_t j = origcolidx[k];
            colidx[rowptr[perm[i]]] = perm[j];
            rowptr[perm[i]]++;
        }
    }
    for (int64_t i = N; i > 0; i--) rowptr[i] = rowptr[i-1];
    rowptr[0] = 0;

    /* if needed, sort the edges of every vertex */
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i = 0; i < N; i++) {
        bool sorted = true;
        for (int64_t k = rowptr[i]+1; k < rowptr[i+1]; k++) {
            if (colidx[k-1] >= colidx[k]) { sorted = false; break; }
        }
        if (sorted) continue;
        for (int64_t k = rowptr[i]+1; k < rowptr[i+1]; k++) {
            int64_t j = colidx[k];
            int64_t l = k-1;
            while (l >= rowptr[i] && colidx[l] > j) {
                colidx[l+1] = colidx[l];
                l--;
            }
            colidx[l+1] = j;
        }
    }
    return 0;
}

/**
 * ‘forwardsolvecsr()’ solves a lower triangular, sparse linear system
 * of equations using forward substitution.
 *
 * The N-by-N matrix L is sparse and lower triangular with its
 * off-diagonal entries provided in compressed sparse row (CSR)
 * format.  In other words, ‘Lrowptr’, ‘Lcolidx’ and ‘L’ are arrays
 * containing row pointers, column offsets and nonzero values of L,
 * respectively. Moreover, ‘Lrowptr’ is an array of length ‘N+1’,
 * whereas ‘Lcolidx’ and ‘L’ are arrays of length ‘Lrowptr[N]’.
 * Finally, the array ‘Ld’ is of length ‘N’ and contains the diagonal
 * entries of the lower triangular matrix L.
 *
 * For L to be lower triangular, it is required that colidx[k] < i for
 * any row i and every k such that rowptr[i] <= k < rowptr[i+1].
 */
int forwardsolvecsr(
    int N,
    const int64_t * rowptr,
    const int64_t * colidx,
    const double * L,
    const double * Ldiag,
    double * x,
    const double * b)
{
    for (int i = 0; i < N; i++) {
        x[i] = b[i];
        for (int k = rowptr[i]; k < rowptr[i+1]; k++) x[i] -= L[k]*x[colidx[k]];
        x[i] /= Ldiag[i];
    }
    return 0;
}

volatile sig_atomic_t upchol_print_progress = 0;
void upcholsighandler(int status)
{
    upchol_print_progress = 1;
}

struct upchol
{
    int64_t N;
    double * b;
    double * x;
    int64_t * xidx;
    int64_t * p;
};

int upchol_init(
    struct upchol * upchol,
    int64_t N,
    bool symbolic)
{
    /* allocate storage for vectors */
    if (!symbolic) {
        double * b = malloc(N * sizeof(double));
        if (!b) return errno;
        double * x = malloc(N * sizeof(double));
        if (!x) { free(b); return errno; }
        int64_t * xidx = malloc(N * sizeof(int64_t));
        if (!xidx) { free(x); free(b); return errno; }
        upchol->N = N;
        upchol->b = b;
        upchol->x = x;
        upchol->xidx = xidx;
    } else {
        int64_t * xidx = malloc(N * sizeof(int64_t));
        if (!xidx) return errno;
        upchol->N = N;
        upchol->b = NULL;
        upchol->x = NULL;
        upchol->xidx = xidx;
    }
    return 0;
}

void upchol_free(
    struct upchol * upchol)
{
    free(upchol->xidx);
    free(upchol->x);
    free(upchol->b);
}

/**
 * ‘upcholsymbfact()’ performs a symbolic factorsiation of a
 * symmetric, positive definite, sparse matrix A using an uplooking
 * Cholesky factorisation.
 *
 * The sparsity pattern of the N-by-N matrix A must be provided with
 * its lower triangular entries in compressed sparse row (CSR) format.
 * Thus, ‘Arowptr’ and ‘Acolidx’ are arrays containing row pointers
 * and column offsets, respectively. Moreover, ‘Arowptr’ is an array
 * of length ‘N+1’, whereas ‘Acolidx’ is an array of length
 * ‘Arowptr[N]’.
 *
 * If successful, the sparsity pattern of the lower triangular
 * Cholesky factor, L, is provided with entries also in CSR format.
 * That is, ‘Lrowptr’ and ‘Lcolidx’ are arrays containing row pointers
 * and column offsets of L, respectively. Again, ‘Lrowptr’ is an array
 * of length ‘N+1’, whereas ‘Lcolidx’ is an array of length
 * ‘Lrowptr[N]’.
 */
static int upcholsymbfact(
    struct upchol * upchol,
    int64_t N,
    int64_t Lmax,
    int64_t * Lsize,
    int64_t * Lrowptr,
    int64_t * Lcolidx,
    const int64_t * Arowptr,
    const int64_t * Acolidx,
    int verbose,
    int progress_interval,
    int64_t * rowsize)
{
    if (N == 0 || *Lsize < 0) return 0;

    struct timespec t0, t1;
    int64_t * xidx = upchol->xidx;

    if (progress_interval > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        upchol_print_progress = 0;
        signal(SIGALRM, upcholsighandler);
        alarm(progress_interval);
    }

    if (*Lsize == 0) (*Lsize)++;
    while (*Lsize < N) {
        int64_t i = *Lsize;

        if (progress_interval > 0 && upchol_print_progress) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "processed %'"PRId64" of %'"PRId64" rows (%4.1f %%) in %'.6f seconds\n",
                    i, N, 100.0*(i/(double)N), timespec_duration(t0, t1));
            upchol_print_progress = 0;
            alarm(progress_interval);
        }

        /*
         * 1. solve the i-by-i lower triangular system
         *
         *    L_{0:i-1;0:i-1} l_{0:i-1;i} = a_{0:i-1;i},
         *
         * where
         *
         *    L_{0:i-1;0:i-1} is the lower triangular matrix
         *    consisting of the first i rows and columns of the
         *    Cholesky factor L.
         *
         *    l_{0:i-1;i} is the solution to the above lower
         *    triangular linear system and a vector containing the
         *    values of the first i rows and the (i+1)-th column of
         *    the Cholesky factor L.
         *
         *    a_{0:i-1;i} is the vector consisting of elements in the
         *    first i rows and the (i+1)-th column of the matrix
         *    A. Alternatively, because the matrix A is symmetric,
         *    this is equivalent to the first i columns of the
         *    (i+1)-th row of A.
         */

        /*
         * Count the number of nonzeros in the solution of the i-by-i
         * lower triangular linear system, and also obtain the
         * positions of those nonzeros.
         */

        int64_t xsize = 0;
        for (int64_t k = Arowptr[i]; k < Arowptr[i+1]; k++) xidx[xsize++] = Acolidx[k];
        for (int64_t ii = 0; ii < i; ii++) {
            int64_t p = 0, q = Lrowptr[ii];
            while (p < xsize && q < Lrowptr[ii+1]) {
                if (xidx[p] < Lcolidx[q]) { p++; }
                else if (xidx[p] > Lcolidx[q]) { q++; }
                else {
                    /* perform a binary search for 'ii' among the
                     * nonzero elements of x */
                    int64_t s = 0, t = xsize-1;
                    while (s <= t) {
                        int64_t r = s + (t-s)/2;
                        if (ii < xidx[r]) { t = r-1; }
                        else if (ii > xidx[r]) { s = r+1; }
                        else { s = t = r; break; }
                    }

                    /* if 'ii' was not found, insert it into the
                     * position found during the binary search */
                    if (s >= xsize || xidx[s] != ii) {
                        for (int64_t l = xsize-1; l >= s; l--) xidx[l+1] = xidx[l];
                        xidx[s] = ii;
                        xsize++;
                    }
                    break;
                }
            }
        }

#ifdef DEBUG
        fprintf(stderr, "i=%"PRId64", xsize=%"PRId64", xidx=[", i, xsize);
        for (int64_t l = 0; l < xsize; l++) fprintf(stderr, " %"PRId64"", xidx[l]);
        fprintf(stderr, "]\n");
#endif

        /*
         * If more space is needed to store the off-diagonal part of
         * the next row of the Cholesky factor L, return an error.
         * This allows the caller to allocate more memory and transfer
         * the partial Cholesky factor to the newly allocated, before
         * calling the function again with to resume the factorisation
         * where we left off.
         */
        if (Lrowptr[i]+xsize > Lmax) {
            if (progress_interval > 0) {
                alarm(0);
                signal(SIGALRM, SIG_DFL);
                upchol_print_progress = 0;
                clock_gettime(CLOCK_MONOTONIC, &t1);
                fprintf(stderr, "processed %'"PRId64" of %'"PRId64" rows (%4.1f %%) in %'.6f seconds\n",
                        i, N, 100.0*(i/(double)N), timespec_duration(t0, t1));
            }
            if (rowsize) *rowsize = xsize;
            return ENOMEM;
        }

        /* add the solution of the i-by-i lower triangular linear
         * system as the (i+1)-th row of the Cholesky factor L */
        Lrowptr[i+1] = Lrowptr[i] + xsize;
        for (int64_t k = 0; k < xsize; k++) Lcolidx[Lrowptr[i]+k] = xidx[k];
        (*Lsize)++;
    }

    if (progress_interval > 0) {
        alarm(0);
        signal(SIGALRM, SIG_DFL);
        upchol_print_progress = 0;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "processed %'"PRId64" of %'"PRId64" rows (%4.1f %%) in %'.6f seconds\n",
                N, N, 100.0, timespec_duration(t0, t1));
    }
    return 0;
}

/**
 * ‘upcholfact()’ performs a Cholesky decomposition of a symmetric,
 * positive definite, sparse matrix A using an uplooking
 * factorisation.
 *
 * The N-by-N matrix A must be provided with its lower triangular
 * entries in compressed sparse row (CSR) format, where ‘Arowptr’,
 * ‘Acolidx’ and ‘A’ are arrays containing row pointers, column
 * offsets and nonzero values, respectively. Moreover, ‘Arowptr’ is an
 * array of length ‘N+1’, whereas ‘Acolidx’ and ‘A’ are arrays of
 * length ‘Arowptr[N]’.  The array ‘Ad’ is of length ‘N’ and contains
 * the diagonal entries of the matrix A.
 *
 * If successful, the lower triangular Cholesky factor, L, is provided
 * with entries also in CSR format. That is, ‘Lrowptr’, ‘Lcolidx’ and
 * ‘L’ are arrays containing row pointers, column offsets and nonzero
 * values of L, respectively. Again, ‘Lrowptr’ is an array of length
 * ‘N+1’, whereas ‘Lcolidx’ and ‘L’ are arrays of length ‘Lrowptr[N]’.
 * The array ‘Ld’ is of length ‘N’ and contains the diagonal entries
 * of the lower triangular matrix L.
 */
static int upcholfact(
    struct upchol * upchol,
    int64_t N,
    int64_t Lmax,
    int64_t * Lsize,
    double * Ld,
    int64_t * Lrowptr,
    int64_t * Lcolidx,
    double * L,
    const double * Ad,
    const int64_t * Arowptr,
    const int64_t * Acolidx,
    const double * A,
    int verbose,
    int progress_interval,
    int64_t * rowsize)
{
    if (N == 0 || *Lsize < 0) return 0;

    struct timespec t0, t1;
    double * b = upchol->b;
    double * x = upchol->x;
    int64_t * xidx = upchol->xidx;

    if (progress_interval > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        upchol_print_progress = 0;
        signal(SIGALRM, upcholsighandler);
        alarm(progress_interval);
    }

    if (*Lsize == 0) {
        Ld[0] = Ad[0];
        if (Ld[0] < 0) return EINVAL;
        Ld[0] = sqrt(Ld[0]);
        (*Lsize)++;
    }

    while (*Lsize < N) {
        int64_t i = *Lsize;

        if (progress_interval > 0 && upchol_print_progress) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "processed %'"PRId64" of %'"PRId64" rows (%4.1f %%) in %'.6f seconds\n",
                    i, N, 100.0*(i/(double)N), timespec_duration(t0, t1));
            upchol_print_progress = 0;
            alarm(progress_interval);
        }

        /*
         * 1. solve the i-by-i lower triangular system
         *
         *    L_{0:i-1;0:i-1} l_{0:i-1;i} = a_{0:i-1;i},
         *
         * where
         *
         *    L_{0:i-1;0:i-1} is the lower triangular matrix
         *    consisting of the first i rows and columns of the
         *    Cholesky factor L.
         *
         *    l_{0:i-1;i} is the solution to the above lower
         *    triangular linear system and a vector containing the
         *    values of the first i rows and the (i+1)-th column of
         *    the Cholesky factor L.
         *
         *    a_{0:i-1;i} is the vector consisting of elements in the
         *    first i rows and the (i+1)-th column of the matrix
         *    A. Alternatively, because the matrix A is symmetric,
         *    this is equivalent to the first i columns of the
         *    (i+1)-th row of A.
         */

        /* Set up right-hand side vector, noting that its nonzero
         * values correspond exactly to nonzeros in the i-th row of
         * the matrix A. */
        for (int64_t j = 0; j < i; j++) b[j] = 0.0;
        for (int64_t k = Arowptr[i]; k < Arowptr[i+1]; k++) b[Acolidx[k]] = A[k];

#ifdef DEBUG
        fprintf(stderr, "b=[");
        for (int64_t j = 0; j < i; j++) fprintf(stderr, " %g", b[j]);
        fprintf(stderr, "]\n");
#endif

        /*
         * Count the number of nonzeros in the solution of the i-by-i
         * lower triangular linear system, and also obtain the
         * positions of those nonzeros.
         */

        int64_t xsize = 0;
        for (int64_t k = Arowptr[i]; k < Arowptr[i+1]; k++) xidx[xsize++] = Acolidx[k];
        for (int64_t ii = 0; ii < i; ii++) {
            int64_t p = 0, q = Lrowptr[ii];
            while (p < xsize && q < Lrowptr[ii+1]) {
                if (xidx[p] < Lcolidx[q]) { p++; }
                else if (xidx[p] > Lcolidx[q]) { q++; }
                else {
                    /* perform a binary search for 'ii' among the
                     * nonzero elements of x */
                    int64_t s = 0, t = xsize-1;
                    while (s <= t) {
                        int64_t r = s + (t-s)/2;
                        if (ii < xidx[r]) { t = r-1; }
                        else if (ii > xidx[r]) { s = r+1; }
                        else { s = t = r; break; }
                    }

                    /* if 'ii' was not found, insert it into the
                     * position found during the binary search */
                    if (s >= xsize || xidx[s] != ii) {
                        for (int64_t l = xsize-1; l >= s; l--) xidx[l+1] = xidx[l];
                        xidx[s] = ii;
                        xsize++;
                    }
                    break;
                }
            }
        }

#ifdef DEBUG
        fprintf(stderr, "i=%"PRId64", xsize=%"PRId64", xidx=[", i, xsize);
        for (int64_t l = 0; l < xsize; l++) fprintf(stderr, " %"PRId64"", xidx[l]);
        fprintf(stderr, "]\n");
#endif

        /*
         * If more space is needed to store the off-diagonal part of
         * the next row of the Cholesky factor L, return an error.
         * This allows the caller to allocate more memory and transfer
         * the partial Cholesky factor to the newly allocated, before
         * calling the function again with to resume the factorisation
         * where we left off.
         */
        if (Lrowptr[i]+xsize > Lmax) {
            if (progress_interval > 0) {
                alarm(0);
                signal(SIGALRM, SIG_DFL);
                upchol_print_progress = 0;
                clock_gettime(CLOCK_MONOTONIC, &t1);
                fprintf(stderr, "processed %'"PRId64" of %'"PRId64" rows (%4.1f %%) in %'.6f seconds\n",
                        i, N, 100.0*(i/(double)N), timespec_duration(t0, t1));
            }
            if (rowsize) *rowsize = xsize;
            return ENOMEM;
        }

        /* solve the i-by-i lower triangular system */
        forwardsolvecsr(i, Lrowptr, Lcolidx, L, Ld, x, b);

#ifdef DEBUG
        fprintf(stderr, "solved %"PRId64"-by-%"PRId64" lower triangular linear system: x=[", i, i);
        for (int64_t j = 0; j < i; j++) fprintf(stderr, " %.4g", x[j]);
        fprintf(stderr, "]\n");
#endif

        /* add the solution of the i-by-i lower triangular linear
         * system as the (i+1)-th row of the Cholesky factor L */
        Lrowptr[i+1] = Lrowptr[i] + xsize;
        for (int64_t k = 0; k < xsize; k++) {
            Lcolidx[Lrowptr[i]+k] = xidx[k];
            L[Lrowptr[i]+k] = x[xidx[k]];
        }

        /*
         * compute the (i+1)-th diagonal entry
         *
         *   l_{i,i} = sqrt(a_{i,i} - l_{1:i-1,i}⋅l_{1:i-1,i})
         */
        Ld[i] = Ad[i];
        for (int64_t k = Lrowptr[i]; k < Lrowptr[i+1]; k++) Ld[i] -= L[k]*L[k];
        if (Ld[i] < 0) return EINVAL;
        Ld[i] = sqrt(Ld[i]);
        (*Lsize)++;
    }

    if (progress_interval > 0) {
        alarm(0);
        signal(SIGALRM, SIG_DFL);
        upchol_print_progress = 0;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "processed %'"PRId64" of %'"PRId64" rows (%4.1f %%) in %'.6f seconds\n",
                N, N, 100.0, timespec_duration(t0, t1));
    }
    return 0;
}

/**
 * `main()`.
 */
int main(int argc, char *argv[])
{
    int err;
    struct timespec t0, t1;
    setlocale(LC_ALL, "");

    /* Set program invocation name. */
    program_invocation_name = argv[0];
    program_invocation_short_name = (
        strrchr(program_invocation_name, '/')
        ? strrchr(program_invocation_name, '/') + 1
        : program_invocation_name);

    /* 1. parse program options */
    struct program_options args;
    err = program_options_init(&args);
    if (err) {
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        return EXIT_FAILURE;
    }

    int nargs = 0, nposargs = 0;
    err = parse_program_options(argc, argv, &args, &nargs, &nposargs);
    if (err) {
        fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
                strerror(err), argv[nargs]);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    if (args.help) {
        program_options_print_help(stdout);
        program_options_free(&args);
        return EXIT_SUCCESS;
    }
    if (args.version) {
        program_options_print_version(stdout);
        program_options_free(&args);
        return EXIT_SUCCESS;
    }
    if (nposargs < 1) {
        program_options_print_usage(stdout);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    /* 2. read the matrix from a Matrix Market file */
    if (args.verbose > 0) {
        fprintf(stderr, "read matrix: ");
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    FILE * f;
    gzFile gzf;
    if (strcmp(args.Apath, "-") == 0) {
        int fd = dup(STDIN_FILENO);
        if (fd == -1) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name, args.Apath, strerror(errno));
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        if (!args.gzip) {
            if ((f = fdopen(fd, "r")) == NULL) {
                if (args.verbose > 0) fprintf(stderr, "\n");
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name, args.Apath, strerror(errno));
                close(fd);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
        } else {
            if ((gzf = gzdopen(fd, "r")) == NULL) {
                if (args.verbose > 0) fprintf(stderr, "\n");
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name, args.Apath, strerror(errno));
                close(fd);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
        }
    } else if (!args.gzip) {
        if ((f = fopen(args.Apath, "r")) == NULL) {
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name, args.Apath, strerror(errno));
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    } else {
        if ((gzf = gzopen(args.Apath, "r")) == NULL) {
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name, args.Apath, strerror(errno));
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    enum mtxobject object;
    enum mtxformat format;
    enum mtxsymmetry symmetry;
    int64_t num_rows;
    int64_t num_columns;
    int64_t num_nonzeros;
    int64_t lines_read = 0;
    int64_t bytes_read = 0;
    err = mtxfile_fread_header(
        &object, &format, &symmetry,
        &num_rows, &num_columns, &num_nonzeros,
        args.gzip, f, gzf, &lines_read, &bytes_read);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                program_invocation_short_name,
                args.Apath, lines_read+1, strerror(err));
        if (args.gzip) gzclose(gzf); else fclose(f);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (num_rows != num_columns || symmetry != mtxsymmetric) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s: %s\n",
                program_invocation_short_name, args.Apath,
                "expected square, symmetric matrix");
        if (args.gzip) gzclose(gzf); else fclose(f);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    int64_t * mtxrowidx = malloc(num_nonzeros * sizeof(int64_t));
    if (!mtxrowidx) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        if (args.gzip) gzclose(gzf); else fclose(f);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int64_t * mtxcolidx = malloc(num_nonzeros * sizeof(int64_t));
    if (!mtxcolidx) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(mtxrowidx);
        if (args.gzip) gzclose(gzf); else fclose(f);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    double * mtxa = malloc(num_nonzeros * sizeof(double));
    if (!mtxa) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(mtxcolidx); free(mtxrowidx);
        if (args.gzip) gzclose(gzf); else fclose(f);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    err = mtxfile_fread_matrix_coordinate_real(
        num_rows, num_columns, num_nonzeros, mtxrowidx, mtxcolidx, mtxa,
        args.gzip, f, gzf, &lines_read, &bytes_read);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                program_invocation_short_name,
                args.Apath, lines_read+1, strerror(err));
        free(mtxa); free(mtxcolidx); free(mtxrowidx);
        if (args.gzip) gzclose(gzf); else fclose(f);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "%'.6f seconds (%'.1f MB/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * bytes_read / timespec_duration(t0, t1));
    }
    if (args.gzip) gzclose(gzf); else fclose(f);

    /* 3. convert to compressed sparse row format. */
    if (args.verbose > 0) {
        fprintf(stderr, "convert to csr: ");
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    int64_t * Arowptr = malloc((num_rows+1) * sizeof(int64_t));
    if (!Arowptr) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(mtxa); free(mtxcolidx); free(mtxrowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int64_t Asize;
    int64_t rowsizemin, rowsizemax;
    int64_t Adiagsize;
    bool symmetric = symmetry == mtxsymmetric;
    bool lower = true;
    bool separate_diagonal = true;
    err = csr_from_coo_size(
        num_rows, num_columns, num_nonzeros, mtxrowidx, mtxcolidx, mtxa,
        Arowptr, &Asize, &rowsizemin, &rowsizemax, &Adiagsize,
        symmetric, lower, separate_diagonal);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(Arowptr);
        free(mtxa); free(mtxcolidx); free(mtxrowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int64_t * Acolidx = malloc(Asize * sizeof(int64_t));
    if (!Acolidx) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(Arowptr);
        free(mtxa); free(mtxcolidx); free(mtxrowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i = 0; i < num_rows; i++) {
        for (int64_t k = Arowptr[i]; k < Arowptr[i+1]; k++)
            Acolidx[k] = 0;
    }
    double * A = malloc(Asize * sizeof(double));
    if (!A) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(Acolidx); free(Arowptr);
        free(mtxa); free(mtxcolidx); free(mtxrowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    double * Ad = malloc(Adiagsize * sizeof(double));
    if (!Ad) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(A); free(Acolidx); free(Arowptr);
        free(mtxa); free(mtxcolidx); free(mtxrowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    if (Adiagsize > 0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int64_t i = 0; i < num_rows; i++) Ad[i] = 0;
    }
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i = 0; i < num_rows; i++) {
        for (int64_t k = Arowptr[i]; k < Arowptr[i+1]; k++)
            A[k] = 0;
    }
    err = csr_from_coo(
        num_rows, num_columns, num_nonzeros, mtxrowidx, mtxcolidx, mtxa,
        Arowptr, Asize, rowsizemin, rowsizemax, Acolidx, A, Ad,
        symmetric, lower, separate_diagonal);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(Ad); free(A); free(Acolidx); free(Arowptr);
        free(mtxa); free(mtxcolidx); free(mtxrowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    free(mtxa); free(mtxcolidx); free(mtxrowidx);

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "%'.6f seconds, %'"PRId64" rows, %'"PRId64" columns, %'"PRId64" nonzeros"
                ", %'"PRId64" to %'"PRId64" nonzeros per row\n",
                timespec_duration(t0, t1), num_rows, num_columns, Asize+Adiagsize, rowsizemin, rowsizemax);
    }

    /* if requested, compute fill-in and exit */
    if (args.fillcount) {
        free(Ad); free(A);

        if (args.verbose > 0) {
            fprintf(stderr, "computing elimination tree: ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        /* compute the elimination tree */
        int64_t * parent = malloc(num_rows * sizeof(int64_t));
        if (!parent) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        int64_t * childptr = malloc((num_rows+1) * sizeof(int64_t));
        if (!childptr) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(parent);
            free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        int64_t * child = malloc(num_rows * sizeof(int64_t));
        if (!child) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(childptr); free(parent);
            free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        etree(num_rows, parent, childptr, child, Arowptr, Acolidx);

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "%'.6f seconds, %'.3f Mnz/s\n",
                    timespec_duration(t0, t1),
                    (double) num_nonzeros * 1e-6 / (double) timespec_duration(t0, t1));
            fprintf(stderr, "post-ordering elimination tree: ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        /* compute a postorder of the elimination tree */
        int64_t * perm = malloc(num_rows * sizeof(int64_t));
        if (!perm) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(child); free(childptr); free(parent);
            free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        int64_t * level = malloc(num_rows * sizeof(int64_t));
        if (!level) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(perm); free(child); free(childptr); free(parent);
            free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        err = postorder(num_rows, perm, level, parent, childptr, child);
        if (err) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
            free(perm); free(child); free(childptr); free(parent);
            free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "%'.6f seconds, %'.3f Mrow/s\n",
                    timespec_duration(t0, t1),
                    (double) num_rows * 1e-6 / (double) timespec_duration(t0, t1));
            fprintf(stderr, "permuting elimination tree based on post-ordering: ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        /* permute the elimination tree according to the
         * post-ordering */
        int64_t * origparent = parent;
        parent = malloc(num_rows * sizeof(int64_t));
        if (!parent) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(perm); free(child); free(childptr); free(parent);
            free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        permute_etree(num_rows, parent, childptr, child, perm, origparent);
        free(origparent);

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "%'.6f seconds, %'.3f Mrow/s\n",
                    timespec_duration(t0, t1),
                    (double) num_rows * 1e-6 / (double) timespec_duration(t0, t1));
            fprintf(stderr, "permuting matrix based on post-ordering: ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        /* permute the rows and columns of the matrix according to the
         * post-ordering of the elimination tree */
        int64_t * Browptr = malloc((num_rows+1) * sizeof(int64_t));
        if (!Browptr) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(perm); free(child); free(childptr); free(parent);
            free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        int64_t * Bcolidx = malloc(Asize * sizeof(int64_t));
        if (!Bcolidx) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(Browptr);
            free(perm); free(child); free(childptr); free(parent);
            free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        permute(num_rows, Browptr, Bcolidx, perm, Arowptr, Acolidx);
        free(perm); free(Acolidx); free(Arowptr);

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "%'.6f seconds, %'.3f Mnz/s\n",
                    timespec_duration(t0, t1),
                    (double) num_nonzeros * 1e-6 / (double) timespec_duration(t0, t1));
        }

        /*
         * compute the row counts; see
         *
         *   John R. Gilbert, Esmond G. Ng, and Barry W. Peyton
         *   (1994). “An Efficient Algorithm to Compute Row and Column
         *   Counts for Sparse Cholesky Factorization.” SIAM Journal
         *   on Matrix Analysis and Applications, vol. 15, no. 4.
         *   DOI: 10.1137/S089547989223692.
         *
         *   Davis, Timothy A., Rajamanickam, Sivasankaran, and
         *   Sid-Lakhdar, Wissam M. (2016). “A survey of direct
         *   methods for sparse linear systems”. Acta numerica,
         *   vol. 25, pp. 383–566. DOI: 10.1017/S0962492916000076.
         */
        if (args.verbose > 0) {
            fprintf(stderr, "computing fill-in: ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int64_t * rowcounts = malloc(num_rows * sizeof(int64_t));
        if (!rowcounts) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(Bcolidx); free(Browptr);
            free(child); free(childptr); free(parent);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        int64_t Ldiagsize = num_rows;
        int64_t Loffdiagsize = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:Loffdiagsize)
#endif
        for (int64_t u = 0; u < num_rows; u++) {
            rowcounts[u] = 1;
            for (int64_t l = Browptr[u]; l < Browptr[u+1]; l++) {
                int64_t p = Bcolidx[l];
                int64_t pp = l < Browptr[u+1]-1 ? Bcolidx[l+1] : u;

                /* find the least common ancestor of p and its
                 * successor pp in the elimination tree. */
                int64_t q = p, r = pp;
                while (q >= 0 && r >= 0 && q != r) {
                    if (q < r) q = parent[q];
                    else r = parent[r];
                }
                rowcounts[u] += level[p];
                if (q >= 0) rowcounts[u] -= level[q];
            }
            Loffdiagsize += rowcounts[u]-1;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "%'.6f seconds, %'.3f Mnz/s, "
                    "%'"PRId64" nonzeros, %'"PRId64" fill-in (nonzeros), "
                    "%'.3f fill-ratio\n",
                    timespec_duration(t0, t1),
                    (double) num_nonzeros * 1e-6 / (double) timespec_duration(t0, t1),
                    Ldiagsize+Loffdiagsize,
                    Loffdiagsize-Asize,
                    Loffdiagsize / (double) Asize);
        }
        free(rowcounts);
        free(Bcolidx); free(Browptr);
        free(level);
        free(child); free(childptr); free(parent);
        program_options_free(&args);
        return EXIT_SUCCESS;
    }

    /* 4. prepare for Cholesky factorisation */
    if (args.verbose > 0) {
        fprintf(stderr, "allocating storage for cholesky factorisation: ");
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    int64_t * Lrowptr = malloc((num_rows+1) * sizeof(int64_t));
    if (!Lrowptr) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(Ad); free(A); free(Acolidx); free(Arowptr);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i = 0; i < num_rows; i++) Lrowptr[i] = 0;
    Lrowptr[num_rows] = 0;

    int64_t Ldiagsize = num_rows;
    int64_t Lallocsize = (int64_t) (args.fillfactor * (double) Asize);
    int Lelemsize = sizeof(int64_t);
    if (!args.symbolic) Lelemsize += sizeof(double);
    int64_t * Lcolidx = malloc(Lallocsize * sizeof(int64_t));
    if (!Lcolidx) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(Lrowptr);
        free(Ad); free(A); free(Acolidx); free(Arowptr);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    for (int64_t i = 0; i < Lallocsize; i++) Lcolidx[i] = 0;
    double * L = NULL;
    double * Ld = NULL;
    if (!args.symbolic) {
        L = malloc(Lallocsize * sizeof(double));
        if (!L) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(Lcolidx); free(Lrowptr);
            free(Ad); free(A); free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        Ld = malloc(Ldiagsize * sizeof(double));
        if (!Ld) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(L); free(Lcolidx); free(Lrowptr);
            free(Ad); free(A); free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int64_t i = 0; i < num_rows; i++) Ld[i] = 0;
    }

    struct upchol upchol;
    err = upchol_init(&upchol, num_rows, args.symbolic);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(Ld); free(L); free(Lcolidx); free(Lrowptr);
        free(Ad); free(A); free(Acolidx); free(Arowptr);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "%'.6f seconds, %'.3f MB initial allocation for cholesky factor (fill-ratio: %'.3f)\n",
                timespec_duration(t0, t1), 1.0e-6*Lallocsize*Lelemsize, args.fillfactor);
    }

    /* 5. perform Cholesky factorisation */
    if (args.verbose > 0 || args.progress_interval > 0) {
        fprintf(stderr, "cholesky factorisation:\n");
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    int64_t i = 0;
    int64_t rowsize;
    int64_t num_flops = 0;
    while (i < num_rows) {
        if (args.symbolic) {
            err = upcholsymbfact(
                &upchol, num_rows, Lallocsize, &i, Lrowptr, Lcolidx,
                Arowptr, Acolidx,
                args.verbose, args.progress_interval, &rowsize);
        } else {
            err = upcholfact(
                &upchol, num_rows, Lallocsize, &i, Ld, Lrowptr, Lcolidx, L,
                Ad, Arowptr, Acolidx, A,
                args.verbose, args.progress_interval, &rowsize);
            if (err == EINVAL) {
                fprintf(stderr, "%s: non-positive definite matrix - "
                        "square root of negative diagonal value %.*g in row %'"PRId64" of cholesky factor\n",
                        program_invocation_short_name, DBL_DIG, Ld[i], i+1);
                upchol_free(&upchol);
                free(Ld); free(L); free(Lcolidx); free(Lrowptr);
                free(Ad); free(A); free(Acolidx); free(Arowptr);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
        }
        if (err == ENOMEM) {
            double Lgrowthfactor = args.growthfactor;
            Lallocsize = (Lallocsize+rowsize) > (Lallocsize*Lgrowthfactor) ? (Lallocsize+rowsize) : (Lallocsize*Lgrowthfactor);
            fprintf(stderr, "growing storage for cholesky factor to %'.3f MB after %'"PRId64" rows\n",
                    1.0e-6 * (double)Lallocsize*Lelemsize, i+1);
            void * tmp = Lcolidx;
            Lcolidx = malloc(Lallocsize * sizeof(int64_t));
            if (!Lcolidx) {
                fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
                upchol_free(&upchol);
                free(Ld); free(L); free(Lcolidx); free(Lrowptr);
                free(Ad); free(A); free(Acolidx); free(Arowptr);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
            memcpy(Lcolidx, tmp, Lrowptr[i]*sizeof(int64_t));
            free(tmp);
            if (!args.symbolic) {
                tmp = L;
                L = malloc(Lallocsize * sizeof(double));
                if (!L) {
                    fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
                    upchol_free(&upchol);
                    free(Ld); free(L); free(Lcolidx); free(Lrowptr);
                    free(Ad); free(A); free(Acolidx); free(Arowptr);
                    program_options_free(&args);
                    return EXIT_FAILURE;
                }
                memcpy(L, tmp, Lrowptr[i]*sizeof(double));
                free(tmp);
            }
        } else if (err) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
            upchol_free(&upchol);
            free(Ld); free(L); free(Lcolidx); free(Lrowptr);
            free(Ad); free(A); free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }
    upchol_free(&upchol);

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "completed cholesky factorisation in "
                "%'.6f seconds, %'.3f Mnz/s, %'.3f Mflop/s,"
                " %'"PRId64" nonzeros, %'"PRId64" fill-in (nonzeros), "
                "%'.3f fill-ratio, %'.3f MB unused\n",
                timespec_duration(t0, t1),
                (double) num_nonzeros * 1e-6 / (double) timespec_duration(t0, t1),
                (double) num_flops * 1e-6 / (double) timespec_duration(t0, t1),
                Ldiagsize+Lrowptr[num_rows],
                Lrowptr[num_rows] - Asize,
                Lrowptr[num_rows] / (double) Asize,
                1.0e-6*(Lallocsize-Lrowptr[num_rows])*Lelemsize);
    }

    /*
     * 6. compare the computed factorisation to the original matrix by
     * calculating the matrix-matrix product B=LL', and then computing
     * the norm ‖B-A‖.
     */
    if (args.verify && !args.symbolic) {
        if (args.verbose > 0 || args.progress_interval > 0) {
            fprintf(stderr, "verifying results - counting nonzeros in B=LL': ");
            if (args.progress_interval > 0) fputc('\n', stderr);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int64_t * Browptr = malloc((num_rows+1) * sizeof(int64_t));
        if (!Browptr) {
            if (args.verbose > 0 && !args.progress_interval) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(Ld); free(L); free(Lcolidx); free(Lrowptr);
            free(Ad); free(A); free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        for (int64_t i = 0; i <= num_rows; i++) Browptr[i] = 0;

        if (args.progress_interval > 0) {
            upchol_print_progress = 0;
            signal(SIGALRM, upcholsighandler);
            alarm(args.progress_interval);
        }

        /* count the number of offdiagonal nonzeros in B=LL' */
        int64_t Bsize = 0;
        for (int64_t i = 0; i < num_rows; i++) {
            if (args.progress_interval > 0 && upchol_print_progress) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                fprintf(stderr, "processed %'"PRId64" of %'"PRId64" rows (%4.1f %%) in %'.6f seconds\n",
                        i, num_rows, 100.0*(i/(double)num_rows), timespec_duration(t0, t1));
                upchol_print_progress = 0;
                alarm(args.progress_interval);
            }

            for (int64_t j = 0; j < i; j++) {
                int64_t p = Lrowptr[i];
                while (p < Lrowptr[i+1] && Lcolidx[p] < j) p++;
                if (p < Lrowptr[i+1] && Lcolidx[p] == j) { Browptr[i+1]++; Bsize++; continue; }
                p = Lrowptr[i];
                int64_t q = Lrowptr[j];
                while (p < Lrowptr[i+1] && q < Lrowptr[j+1]) {
                    if (Lcolidx[p] < Lcolidx[q]) p++;
                    else if (Lcolidx[p] > Lcolidx[q]) q++;
                    else { Browptr[i+1]++; Bsize++; break; }
                }
            }
        }
        for (int64_t i = 0; i < num_rows; i++) Browptr[i+1] += Browptr[i];

        if (args.progress_interval > 0) {
            alarm(0);
            signal(SIGALRM, SIG_DFL);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "processed %'"PRId64" of %'"PRId64" rows (%4.1f %%) in %'.6f seconds\n",
                    num_rows, num_rows, 100.0, timespec_duration(t0, t1));
            upchol_print_progress = 0;
        }

        int64_t * Bcolidx = malloc(Bsize * sizeof(int64_t));
        if (!Bcolidx) {
            if (args.verbose > 0 && !args.progress_interval) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(Browptr);
            free(Ld); free(L); free(Lcolidx); free(Lrowptr);
            free(Ad); free(A); free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        double * B = malloc(Bsize * sizeof(double));
        if (!B) {
            if (args.verbose > 0 && !args.progress_interval) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(Bcolidx); free(Browptr);
            free(Ld); free(L); free(Lcolidx); free(Lrowptr);
            free(Ad); free(A); free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        int64_t Bdiagsize = num_rows;
        double * Bd = malloc(Bdiagsize * sizeof(double));
        if (!Bd) {
            if (args.verbose > 0 && !args.progress_interval) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(B); free(Bcolidx); free(Browptr);
            free(Ld); free(L); free(Lcolidx); free(Lrowptr);
            free(Ad); free(A); free(Acolidx); free(Arowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int64_t i = 0; i < num_rows; i++) {
            for (int64_t k = Browptr[i]; k < Browptr[i+1]; k++)
                B[k] = 0;
        }
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int64_t i = 0; i < num_rows; i++) Bd[i] = 0;

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            if (args.progress_interval > 0) fprintf(stderr, "done counting nonzeros in B=LL' in ");
            fprintf(stderr, "%'.6f seconds\n", timespec_duration(t0, t1));
        }

        if (args.verbose > 0 || args.progress_interval > 0) {
            fprintf(stderr, "verifying results - computing column offsets in B=LL': ");
            if (args.progress_interval > 0) fputc('\n', stderr);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        if (args.progress_interval > 0) {
            upchol_print_progress = 0;
            signal(SIGALRM, upcholsighandler);
            alarm(args.progress_interval);
        }

        /* compute column offsets for offdiagonal nonzeros in B=LL' */
        for (int64_t i = 0; i < num_rows; i++) {
            if (args.progress_interval > 0 && upchol_print_progress) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                fprintf(stderr, "processed %'"PRId64" of %'"PRId64" rows (%4.1f %%) in %'.6f seconds\n",
                        i, num_rows, 100.0*(i/(double)num_rows), timespec_duration(t0, t1));
                upchol_print_progress = 0;
                alarm(args.progress_interval);
            }

            for (int64_t j = 0; j < i; j++) {
                int64_t p = Lrowptr[i];
                while (p < Lrowptr[i+1] && Lcolidx[p] < j) p++;
                if (p < Lrowptr[i+1] && Lcolidx[p] == j) {
                    Bcolidx[Browptr[i]++] = j;
                    continue;
                }
                p = Lrowptr[i];
                int64_t q = Lrowptr[j];
                while (p < Lrowptr[i+1] && q < Lrowptr[j+1]) {
                    if (Lcolidx[p] < Lcolidx[q]) p++;
                    else if (Lcolidx[p] > Lcolidx[q]) q++;
                    else { Bcolidx[Browptr[i]++] = j; break; }
                }
            }
        }
        for (int i = num_rows; i > 0; i--) Browptr[i] = Browptr[i-1];
        Browptr[0] = 0;

        if (args.progress_interval > 0) {
            alarm(0);
            signal(SIGALRM, SIG_DFL);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "processed %'"PRId64" of %'"PRId64" rows (%4.1f %%) in %'.6f seconds\n",
                    num_rows, num_rows, 100.0, timespec_duration(t0, t1));
            upchol_print_progress = 0;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            if (args.progress_interval > 0) fprintf(stderr, "done computing column offsets in B=LL' in ");
            fprintf(stderr, "%'.6f seconds\n", timespec_duration(t0, t1));
        }

        if (args.verbose > 0 || args.progress_interval > 0) {
            fprintf(stderr, "verifying results - computing B=LL': ");
            if (args.progress_interval > 0) fputc('\n', stderr);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        if (args.progress_interval > 0) {
            upchol_print_progress = 0;
            signal(SIGALRM, upcholsighandler);
            alarm(args.progress_interval);
        }

        /* compute matrix-vector product B=LL' */
        for (int64_t i = 0; i < num_rows; i++) {
            if (args.progress_interval > 0 && upchol_print_progress) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                fprintf(stderr, "processed %'"PRId64" of %'"PRId64" rows (%4.1f %%) in %'.6f seconds\n",
                        i, num_rows, 100.0*(i/(double)num_rows), timespec_duration(t0, t1));
                upchol_print_progress = 0;
                alarm(args.progress_interval);
            }

            for (int64_t j = 0; j < i; j++) {
                int64_t p = Lrowptr[i];
                while (p < Lrowptr[i+1] && Lcolidx[p] < j) p++;
                if (p < Lrowptr[i+1] && Lcolidx[p] == j) {
                    int64_t r = Browptr[i];
                    while (r < Browptr[i+1] && Bcolidx[r] != j) r++;
                    B[r] += L[p]*Ld[j];
                }
                p = Lrowptr[i];
                int64_t q = Lrowptr[j];
                while (p < Lrowptr[i+1] && q < Lrowptr[j+1]) {
                    if (Lcolidx[p] < Lcolidx[q]) p++;
                    else if (Lcolidx[p] > Lcolidx[q]) q++;
                    else {
                        int64_t r = Browptr[i];
                        while (r < Browptr[i+1] && Bcolidx[r] != j) r++;
                        B[r] += L[p]*L[q];
                        p++; q++;
                    }
                }
            }
            for (int64_t p = Lrowptr[i]; p < Lrowptr[i+1]; p++) Bd[i] += L[p]*L[p];
            Bd[i] += Ld[i]*Ld[i];
        }

        if (args.progress_interval > 0) {
            alarm(0);
            signal(SIGALRM, SIG_DFL);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "processed %'"PRId64" of %'"PRId64" rows (%4.1f %%) in %'.6f seconds\n",
                    num_rows, num_rows, 100.0, timespec_duration(t0, t1));
            upchol_print_progress = 0;
        }

/* #ifdef DEBUG */
/*         fprintf(stdout, "%%%%MatrixMarket matrix coordinate real symmetric\n"); */
/*         fprintf(stdout, "%"PRId64" %"PRId64" %"PRId64"\n", num_rows, num_columns, Bdiagsize+Bsize); */
/*         for (int64_t i = 0; i < num_rows; i++) { */
/*             int64_t k = Browptr[i]; */
/*             while (k < Browptr[i+1] && Bcolidx[k] < i) { */
/*                 fprintf(stdout, "%"PRId64" %"PRId64" %.*g\n", i+1, Bcolidx[k]+1, DBL_DIG, B[k]); */
/*                 k++; */
/*             } */
/*             fprintf(stdout, "%"PRId64" %"PRId64" %.*g\n", i+1, i+1, DBL_DIG, Bd[i]); */
/*             while (k < Browptr[i+1]) { */
/*                 fprintf(stdout, "%"PRId64" %"PRId64" %.*g\n", i+1, Bcolidx[k]+1, DBL_DIG, B[k]); */
/*                 k++; */
/*             } */
/*         } */
/* #endif */

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            if (args.progress_interval > 0) fprintf(stderr, "done computing B=LL' in ");
            fprintf(stderr, "%'.6f seconds\n", timespec_duration(t0, t1));
        }

        if (args.verbose > 0 || args.progress_interval > 0) {
            fprintf(stderr, "verifying results - computing error: ");
            if (args.progress_interval > 0) fputc('\n', stderr);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        if (args.progress_interval > 0) {
            upchol_print_progress = 0;
            signal(SIGALRM, upcholsighandler);
            alarm(args.progress_interval);
        }

        /*
         * Compute the difference B-A and its max- and Frobenius norm.
         * Note that the sparsity pattern of A is a subset of the
         * sparsity pattern of B.
         */
        double diffnrmmax = 0, diffnrm2 = 0;
        for (int64_t i = 0; i < num_rows; i++) {
            if (args.progress_interval > 0 && upchol_print_progress) {
                fprintf(stderr, "\n%'"PRId64" of %'"PRId64" rows (%4.1f %%)\n",
                        i, num_rows, 100.0*(i/(double)num_rows));
                upchol_print_progress = 0;
                alarm(args.progress_interval);
            }

            int64_t p = Arowptr[i], q = Browptr[i];
            while (p < Arowptr[i+1] && q < Browptr[i+1]) {
                if (Acolidx[p] < Bcolidx[q]) {
#ifdef DEBUG
                    fprintf(stderr, "warning: nonzero found in A at row %'"PRId64" and column %'"PRId64", but not found in B!\n",
                            i, Acolidx[p]);
#endif
                    if (diffnrmmax < fabs(A[p])) diffnrmmax = fabs(A[p]);
                    diffnrm2 += A[p]*A[p];
                    p++;
                } else if (Acolidx[p] > Bcolidx[q]) {
                    if (diffnrmmax < fabs(B[q])) diffnrmmax = fabs(B[q]);
                    diffnrm2 += B[q]*B[q];
                    q++;
                } else {
                    if (diffnrmmax < fabs(A[p]-B[q])) diffnrmmax = fabs(A[p]-B[q]);
                    diffnrm2 += (A[p]-B[q])*(A[p]-B[q]);
                    p++; q++;
                }
            }
            if (diffnrmmax < fabs(Ad[i]-Bd[i])) diffnrmmax = fabs(Ad[i]-Bd[i]);
            diffnrm2 += (Ad[i]-Bd[i])*(Ad[i]-Bd[i]);
        }

        if (args.progress_interval > 0) {
            alarm(0);
            signal(SIGALRM, SIG_DFL);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "processed %'"PRId64" of %'"PRId64" rows (%4.1f %%) in %'.6f seconds\n",
                    i, num_rows, 100.0*(i/(double)num_rows), timespec_duration(t0, t1));
            upchol_print_progress = 0;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            if (args.progress_interval > 0) fprintf(stderr, "done computing error in ");
            fprintf(stderr, "%'.6f seconds, ", timespec_duration(t0, t1));
        }
        fprintf(stderr, "%.*g max-norm error, %.*g Frobenius-norm error\n",
                DBL_DIG, diffnrmmax, DBL_DIG, diffnrm2);

        free(Bd); free(B); free(Bcolidx); free(Browptr);
    }

    free(Ad); free(A); free(Acolidx); free(Arowptr);

    /* 7. write the Cholesky factor to a Matrix Market file */
    if (!args.quiet) {
        if (args.verbose > 0) {
            fprintf(stderr, "mtxfile_write: ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        if (!args.symbolic) {
            fprintf(stdout, "%%%%MatrixMarket matrix coordinate real general\n");
            fprintf(stdout, "%"PRId64" %"PRId64" %"PRId64"\n", num_rows, num_columns, Ldiagsize+Lrowptr[num_rows]);
            for (int64_t i = 0; i < num_rows; i++) {
                int64_t k = Lrowptr[i];
                while (k < Lrowptr[i+1] && Lcolidx[k] < i) {
                    fprintf(stdout, "%"PRId64" %"PRId64" %.*g\n", i+1, Lcolidx[k]+1, DBL_DIG, L[k]);
                    k++;
                }
                fprintf(stdout, "%"PRId64" %"PRId64" %.*g\n", i+1, i+1, DBL_DIG, Ld[i]);
                while (k < Lrowptr[i+1]) {
                    fprintf(stdout, "%"PRId64" %"PRId64" %.*g\n", i+1, Lcolidx[k]+1, DBL_DIG, L[k]);
                    k++;
                }
            }
        } else {
            fprintf(stdout, "%%%%MatrixMarket matrix coordinate pattern general\n");
            fprintf(stdout, "%"PRId64" %"PRId64" %"PRId64"\n", num_rows, num_columns, Ldiagsize+Lrowptr[num_rows]);
            for (int64_t i = 0; i < num_rows; i++) {
                int64_t k = Lrowptr[i];
                while (k < Lrowptr[i+1] && Lcolidx[k] < i) {
                    fprintf(stdout, "%"PRId64" %"PRId64"\n", i+1, Lcolidx[k]+1);
                    k++;
                }
                fprintf(stdout, "%"PRId64" %"PRId64"\n", i+1, i+1);
                while (k < Lrowptr[i+1]) {
                    fprintf(stdout, "%"PRId64" %"PRId64"\n", i+1, Lcolidx[k]+1);
                    k++;
                }
            }
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "%'.6f seconds (%'.3f Mnz/s)\n",
                    timespec_duration(t0, t1),
                    1e-6*(Ldiagsize+Lrowptr[num_rows])/(double)timespec_duration(t0, t1));
        }
    }

    free(Ld); free(L); free(Lcolidx); free(Lrowptr);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
