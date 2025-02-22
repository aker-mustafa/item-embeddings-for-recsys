﻿using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace ImageToSRS {
    class Program {
        static void Main(string[] args) {
            var inMemoryList = new List<List<int>>();
            var itemList = new SortedList<int, int>();
            //Read data
            using (var file = new StreamReader(args[0])) {
                string line;
                while ((line = file.ReadLine()) != null) {
                    var lineList = new List<int>();
                    var array = line.Split(',');
                    for (int i = 0; i < array.Length; i++) {
                        if (string.IsNullOrEmpty(array[i]))
                            break;
                        int item = Convert.ToInt32(array[i].Split('.')[0]);
                        if (!itemList.ContainsKey(item))
                            itemList.Add(item, 1);
                        else
                            itemList[item]++;
                        lineList.Add(item);
                    }
                    inMemoryList.Add(lineList);
                }
            }
            //Create adjacency matrix
            var masterDict = new SortedList<int, SortedList<int, int?>>();
            foreach (var item in itemList) {
                masterDict.Add(item.Key, new SortedList<int, int?>());
            }
            foreach (var row in masterDict) {
                foreach (var session in inMemoryList) {
                    if (session.Contains(row.Key)) {
                        foreach (var item in session) {
                            if (row.Value.ContainsKey(item))
                                row.Value[item]++;
                            else
                                row.Value.Add(item, 1);
                        }
                    }
                }
            }
            //Write matrix to file
            var outputTime = DateTime.Now;
            using (var file = new StreamWriter($"featureSet_full_yoochoose_{outputTime:yyyy-MM-dd_hh_mm_ss}.csv")) {
                var line = "," + string.Join(',', itemList.Keys);
                file.WriteLine(line);
                while (masterDict.Count > 0) {
                    var items = masterDict.First().Value;
                    foreach (var item in itemList) {
                        if (!items.ContainsKey(item.Key))
                            items.Add(item.Key, null);
                    }
                    line = masterDict.First().Key.ToString() + "," + string.Join(',', items.Values);
                    file.WriteLine(line);
                    masterDict.RemoveAt(0);
                }
            }
        }
    }
}
