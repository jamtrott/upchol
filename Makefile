# Benchmark for sparse Cholesky factorisation
#
# Copyright (C) 2023 James D. Trotter
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <https://www.gnu.org/licenses/>.
#
# Authors: James D. Trotter <james@simula.no>
#
# Last modified: 2023-02-25
#
# Benchmark program for sparse Cholesky factorization.

upchol = upchol

all: $(upchol)
clean:
	rm -f $(upchol_c_objects) $(upchol)
.PHONY: all clean

CFLAGS += -g -Wall

ifndef NO_OPENMP
CFLAGS += -fopenmp -DWITH_OPENMP
endif

upchol_c_sources = upchol.c
upchol_c_headers =
upchol_c_objects := $(foreach x,$(upchol_c_sources),$(x:.c=.o))
$(upchol_c_objects): %.o: %.c $(upchol_c_headers)
	$(CC) -c $(CFLAGS) $< -o $@
$(upchol): $(upchol_c_objects)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lm -o $@
