
Gridded Population of the World Version 3 (GPWv3): Population Grids.

SUGGESTED CITATION

Center for International Earth Science Information Network (CIESIN), Columbia 
University; and Centro Internacional de Agricultura Tropical (CIAT). 2005. 
Gridded Population of the World Version 3 (GPWv3): Population Grids. Palisades, 
NY: Socioeconomic Data and Applications Center (SEDAC), Columbia University. 
Available at http://sedac.ciesin.columbia.edu/gpw. (date of download).

ARCHIVE CONTENTS

The data contained in this archive are the final version of GPWv3, and
should be used instead of the previous (alpha, beta) versions.

This archive contains population counts, both UN-adjusted and unadjusted, 
in BIL format. The raster data are at 2.5 arc-minutes resolution and contain 
the following data:

p90g	population counts in 1990, unadjusted
p90ag	population counts in 1990, adjusted to match UN totals
p95g	population counts in 1995, unadjusted
p95ag	population counts in 1995, adjusted to match UN totals
p00g	population counts in 2000, unadjusted
p00ag	population counts in 2000, adjusted to match UN totals

The BIL format stores the data as binary files with associated 
information stored in world, header, and statistics files. The set 
of BIL files and what they contain are described below.

.bil	Binary format data

.blw	Resolution information with the following lines:
		resolution in x-dimension
		rotation factor (always zero for GPW)
		rotation factor (always zero for GPW)
		resolution in y-dimension
		x-coordinate of the center of the upper-left cell
		y-coordinate of the center of the upper-left cell

.hdr	Header file with the following lines:
		Byte order (motorola or intel, M or I)
		Layout (BIL or Band Interleaved by Line)
		NROWS (number of rows)
		NCOLS (number of columns)
		NBANDS (number of bands)
		NBITS (number of bits -- 4, 8, 24, or 32)
		BANDROWBYTES (number of byes per band, per row)
		TOTALROWBYES (total byes of data per row)
		BANDGAPBYTES (Does not apply for BIL)

.stx	Statistics of the grid before export, the grid statistics are stored
	on one line. Note that the values (listed below) are for the grid and therefore
	they exclude nodata. Conversion may set nodata areas to zero, which will 
	alter the statistics.
		Band (always one)
		Minimum value
		Maximum value
		Mean value
		Standard deviation.

The data are stored in geographic coordinates of decimal degrees based 
on the World Geodetic System spheroid of 1984 (WGS84).

USE CONSTRAINTS:

The Trustees of Columbia University in the City of New York and the Centro 
Internacional de Agricultura Tropical (CIAT) hold the copyright of this dataset. 
Users are prohibited from any commercial, non-free resale, or redistribution 
without explicit written permission from CIESIN or CIAT. Users should acknowledge 
CIESIN and CIAT as the source used in the creation of any reports, publications, 
new data sets, derived products, or services resulting from the use of this data 
set. CIESIN and CIAT also request reprints of any publications and notification 
of any redistributing efforts.

DISTRIBUTION LIABILITY

CIESIN follows procedures designed to ensure that data disseminated by CIESIN 
are of reasonable quality. If, despite these procedures, users encounter apparent 
errors or misstatements in the data, they should contact SEDAC User Services at 
+1 845-365-8920 or via email at ciesin.info@ciesin.columbia.edu. Neither CIESIN nor 
NASA verifies or guarantees the accuracy, reliability, or completeness of any data 
provided. CIESIN provides this data without warranty of any kind whatsoever, either 
express or implied. CIESIN shall not be liable for incidental, consequential, or 
special damages arising out of the use of any data provided by CIESIN.

ACKNOWLEDGEMENTS

This work, including access to the data and technical assistance, is provided by 
CIESIN, with funding from the National Aeronautics and Space Administration under 
Contract NAS5-03117 for the Continued Operation of the Socioeconomic Data and 
Applications Center (SEDAC).