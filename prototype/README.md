# Krah

Tools to visualize and extract data from Krah pipe production logs

## Modules

* toru.py - command line interface
* vis.py - visualization with matplotlib
* data.py - data loading and extraction
* log.py - utilities to transform and analyze attributes and features from the production log
* qc.py - utilities to transform and analyze QC measurements data
* model.py - estimation of things that are not in raw data (like thickness between ribs)
* ds_prep.py - copy and filter pipe data to prepare datasets

## Usage

Pipe data visualizer

```
usage: toru.py [-h] [-p [list of pipe ids ...]] [-f [list of pipe ids ...]] [-l [list of pipe ids ...]] [-m mode] [-F feature]
               [-A attribute] [-M model] [-X x-axis] [-S segment] [-P peak mode] [-s aggregate mode] [-e measurement] [-d detection mode] [-b base directory]
               [-i input file] [-o output file] [-w filter width] [-y ymin ymax]

options:
  -h, --help  show this help message and exit
  -p          show only pipe ids, use space as separator (31052 31053 31127), defaults to all
  -f          filter pipe ids (as above), defaults to none
  -l          highlight pipe ids (as above), defaults to none
  -m          run mode (plot,segment,size,aggr,shape,feature,shape_regr,scatter,ts_classif,ts_regr)
  -F          feature name
  -A          attribute name
  -M          model name
  -X          X-axis attribute name
  -S          select segment (0, 1, 2, default: none)
  -P          show peaks (upper/lower/none), default none
  -s          aggregate function (sum/mean/median)
  -e          measurement number
  -d          end detection
  -b          base directory
  -i          input file (pipe metadata)
  -o          output file
  -w          pulse filter width
  -y          y range (-y min max)
  -t min_len  truncate timeseries to same length (ts_classif, ts_regr modes).
              Argument is min length
```

### Run modes

* `plot` - plot an attribute (-A name), feature (-F name) or model estimate (-M name) from the production log
* `shape` - plot an attribute (-A name) or model estimate (-M name) from the QC measurements
* `size` - estimate pipe length
* `segment` - visualize pipe segments from the QC measurements
* `aggr` - aggregate bar plot from production log (supports -A, -F, -M), use -s to select the aggregate function
* `feature` - feature extraction, predicted value is class (scrap yes/no). Use -o output file. If -o is omitted, plots 5 best predictors
* `shape_aggr` - feature extraction, predicted value is some regression target from QC data. Use -o output file. Use -A, -M, -s to select the target and the aggregate function
* `scatter` - pairwise plots of attributes and features
* `ts_classif` - time series extraction, predicted value is class (scrap yes/no). Use -o output file.
* `ts_regr` - time series extraction, predicted value is some regression target from QC data. Use -o output file. Use -A, -M, -s to select the target and the aggregate function. Use -t min_length to truncate series to same length

### Attributes

Anything from production log ("CarriageSpeed") or QC measurements ("gThickness").

### Feature names

Combined attributes

* ExtruderTotal - total output of extruders 1, 2, 3
* SurfaceStep - how fast the surface moves. Affects coretube spacing
* CoretubeStretch - how much the coretube is stretched
* MaterialStretch - higher value means thinner deposit on tube

There are other experimental features.

### Models

* deposit - estimate material deposited on the pipe (production log)
* thickness - thickness of the tube, excluding ribs and socket/spigot
* min_thickness - cleaned thickness model, keeps only lowest points
* min_height - lowest points of ribs height
* ribs - distance between ribs

Filtering script

```
usage: ds_prep.py [-h] [-O output directory] [-q query] dir1 dir2 ...

options:
  -h, --help  show this help message and exit
  -O dirname  output directory
  -q query    query expression (pandas syntax, use column names from QRS
              report)
```

## Examples

Plot thickness profile for one pipe. The -i option is the production materials
use file that contains references to both the production log and QC measurements.
The measurement defaults to 0.

```
./toru.py -b '../AIRE & TalTech andmed' -m shape -A gThickness -i export_merged2.csv -p 31050
```

Plot the shape of the spigot (segment 2). Color is the class (scrap yes/no) from
the input file. One pipe is filtered out.

```
./toru.py -b '../AIRE & TalTech andmed' -m shape -A gThickness -i export_merged2.csv -S 2 -f 31050
```

Plot some attribute (CarriageSpeed) from production for all pipes. Period
shown is when the extruders are working (-d extruder).

```
./toru.py -b '../PR90-011.57 RG' -m plot -A CarriageSpeed -i export_merged.csv -d extruder
```

Plot an attribute for selected pipes, highlighting one pipe. Period
shown is when the coretube is producted (-d coretube).

```
./toru.py -b '../PR90-011.57 RG' -m plot -A 'ToolTemperatures[9]' -i export_merged_h_madal.csv -p 31052 31053 -l 31053 -d coretube
```

Bar plot of a production log features. Color is the class of the pipe (correct rib spacing or not)
from the input file.

```
./toru.py -b '../PR90-011.57 RG' -m aggr -s median -F SurfaceStep -d coretube -i export_merged_ribivahe.csv
```

Save tabular data to a file. Extracts aggregate features from production log. Adds a prediction
target (scrap y/n) from the input file.

```
./toru.py -b '../PR90-011.57 RG' -m feature -s median -i export_merged_h_madal.csv -d coretube -o '../test.csv'
```

Pairwise plots of a given feature against all other features.

```
./toru.py -b '../PR90-011.57 RG' -m scatter -s median -i export_merged_h_madal.csv -d coretube
```

Extract timeseries regression data (minimum profile height).

```
./toru.py -b '../PR90-011.57 RG' -m ts_regr -M min_height -i export_autogen.csv -o ../test10.csv -t 900 -S 1
```

Copy pipes from multiple directories, profile "PR", length 6m

```
./ds_prep.py -O ../copy_test -q '`Tellitud pikkus` == 6000 and `Profiili tüüp` == \"PR\"' '../AIRE & TalTech andmed/20230104 QRS report.xlsx' '../PR90-011.57 RG/QRS report.xlsx' '../Tellimus 20230337/20230337 QRS report.xlsx' '../07-08.2023/QRS raport.xlsx'
```