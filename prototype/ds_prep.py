#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os.path
import data
import shutil
#import numpy as np

QRS_FIELDS = ["Diameeter",
              "Korrutoru Diameeter",
              "Välisseina paksus 1.1",
              "Nõutav välisseina paksus"]

class PipeFilter:
    def __init__(self, filenames, query=""):
        self.base_dirs = [os.path.split(fn) for fn in filenames]
        self.query = query
        self.stats = {
            "log_missing" : 0,
            "log_multi" : 0,
            "qc_missing" : 0,
            }

    def load_qrs(self, base_dir, qrs_file):
        fn = os.path.join(base_dir, qrs_file)
        return pd.read_excel(fn, header=0)

    def load_export(self, base_dir):
        dirname = os.path.join(base_dir, data.EXP_DIR)
        export_data = {}
        for fn in os.listdir(dirname):
            if os.path.splitext(fn)[1] == ".csv":
                export_data = self.merge_export_file(export_data,
                    os.path.join(dirname, fn))
        return export_data

    def merge_export_file(self, export_data, fn):
        df = pd.read_csv(fn, header=0)
        for _, row in df.iterrows():
            pipe_id = row["QRS Pipe ID"]
            if pipe_id not in export_data:
                export_data[pipe_id] = {
                    "guid": [],
                    "name": [],
                    }
            guid = row["PipeData GUID"]
            name = row["Name"]
            if guid not in export_data[pipe_id]["guid"]:
                export_data[pipe_id]["guid"].append(guid)
                export_data[pipe_id]["name"].append(name)
        return export_data

    def filt(self, qrs, export_data):
        if self.query:
            qrs = qrs.query(self.query)
        for _, row in qrs.iterrows():
            pipe_id = row["ID"]
            if pipe_id in export_data:
                guids = export_data[pipe_id]["guid"]
                names = export_data[pipe_id]["name"]
                if len(guids) < 1:
                    self.stats["log_missing"] += 1
                elif len(guids) > 1:
                    self.stats["log_multi"] += 1
                else:
                    rec = {
                        "PipeData GUID": guids[0],
                        "Name": names[0],
                        "Is Scrap": 0,
                        "QRS Pipe ID": pipe_id}
                    for fld in QRS_FIELDS:
                        rec[fld] = row[fld]
                    yield pipe_id, rec

    def pipe_files(self, base_dir, pipe_id, export_meta):
        log_fn = os.path.join(base_dir, data.LOG_DIR,
                              export_meta["PipeData GUID"] + ".csv")
        if not os.path.isfile(log_fn):
            self.stats["log_missing"] += 1
            return None
        qc_fns = []
        for scan in [0, 1, 2]:
            scan_fn = os.path.join(base_dir, data.QC_DIR,
                                    "id{}_{}.csv".format(pipe_id, scan))
            if not os.path.isfile(log_fn):
                self.stats["qc_missing"] += 1
                return None
            qc_fns.append(scan_fn)
        return {"log": log_fn, "qc": qc_fns}

    def get_pipes(self):
        seen = set()
        for base_dir, qrs_file in self.base_dirs:
            qrs = self.load_qrs(base_dir, qrs_file)
            #print(qrs.shape)
            export_data = self.load_export(base_dir)
            for pipe_id, export_meta in self.filt(qrs, export_data):
                file_meta = self.pipe_files(base_dir, pipe_id, export_meta)
                if file_meta and pipe_id not in seen:
                    seen.add(pipe_id)
                    yield pipe_id, export_meta, file_meta

EXP_FILE = "export_autogen.csv"
class PipeCopier:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.stats = {"copy_failed" : 0}

    def make_dirs(self):
        if os.path.exists(self.base_dir):
            raise ValueError("{} already exists, not overwriting".format(
                self.base_dir))
        os.mkdir(self.base_dir)
        os.mkdir(os.path.join(self.base_dir, data.EXP_DIR))
        os.mkdir(os.path.join(self.base_dir, data.LOG_DIR))
        os.mkdir(os.path.join(self.base_dir, data.QC_DIR))

    def copy(self, pipe_gen):
        self.make_dirs()
        export_data = []
        for pipe_id, export_meta, file_meta in pipe_gen:
            try:
                shutil.copy(file_meta["log"],
                            os.path.join(self.base_dir, data.LOG_DIR))
                for fn in file_meta["qc"]:
                    shutil.copy(fn,
                                os.path.join(self.base_dir, data.QC_DIR))
            except:
                self.stats["copy_failed"] += 1
                continue
            export_data.append(export_meta)
        df = pd.DataFrame.from_records(export_data)
        df.to_csv(os.path.join(self.base_dir, data.EXP_DIR, EXP_FILE),
                  encoding='utf-8', index=False)
        return df.shape[0]

# args="'../AIRE & TalTech andmed/20230104 QRS report.xlsx'"
# args="'../07-08.2023/QRS raport.xslx'"
# args="'../AIRE & TalTech andmed/20230104 QRS report.xlsx' '../PR90-011.57 RG/QRS report.xlsx' '../Tellimus 20230337/20230337 QRS report.xlsx' '../07-08.2023/QRS raport.xlsx'"
# args="-O ../copy_test '../AIRE & TalTech andmed/20230104 QRS report.xlsx' '../PR90-011.57 RG/QRS report.xlsx'"
# args="-q '`Tellitud pikkus` == 6000' '../AIRE & TalTech andmed/20230104 QRS report.xlsx' '../PR90-011.57 RG/QRS report.xlsx'"

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    # p.add_argument('-d', action="store_true",
    #     help="dry run")
    p.add_argument('-O', type=str, default="",
        help="output directory")
    p.add_argument('-q', type=str, default="",
        help="query expression (pandas syntax, use column names from QRS report)")
    p.add_argument('filenames', nargs=argparse.REMAINDER)
    args = p.parse_args()

    if args.filenames:
        ps = PipeFilter(args.filenames, args.q)
        if args.O:
            cp = PipeCopier(args.O)
            cnt = cp.copy(ps.get_pipes())
            print("Copy stats", cp.stats)
        else:
            cnt = 0
            for _ in ps.get_pipes():
                cnt += 1

        print("Filter stats", ps.stats)
        print("{} pipes".format(cnt))
