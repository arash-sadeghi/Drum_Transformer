# Transformer for Drum

## tokenization observation:

- Original midi:
```
[INFO] 0 : program_change channel=9 program=0 time=0
[INFO] 1 : program_change channel=9 program=0 time=0
[INFO] 2 : note_on channel=9 note=38 velocity=7 time=2.115625
[INFO] 3 : note_on channel=9 note=38 velocity=0 time=0.060937500000000006
[INFO] 4 : note_on channel=9 note=38 velocity=10 time=0
[INFO] 5 : note_on channel=9 note=38 velocity=0 time=0.0390625
[INFO] 6 : control_change channel=9 control=4 value=90 time=0.003125
[INFO] 7 : control_change channel=9 control=4 value=88 time=0.046875
[INFO] 8 : control_change channel=9 control=4 value=86 time=0.046875
[INFO] 9 : control_change channel=9 control=4 value=84 time=0.046875
[INFO] 10 : control_change channel=9 control=4 value=81 time=0.046875
[INFO] 11 : control_change channel=9 control=4 value=79 time=0.046875
[INFO] 12 : note_on channel=9 note=38 velocity=56 time=0.0015625
[INFO] 13 : control_change channel=9 control=4 value=77 time=0.045312500000000006
[INFO] 14 : control_change channel=9 control=4 value=75 time=0.021875000000000002
[INFO] 15 : control_change channel=9 control=4 value=46 time=0.0234375
[INFO] 16 : note_on channel=9 note=38 velocity=0 time=0.010937500000000001
[INFO] 17 : control_change channel=9 control=4 value=17 time=0.0125
[INFO] 18 : control_change channel=9 control=4 value=4 time=0.010937500000000001
[INFO] 19 : control_change channel=9 control=4 value=34 time=0.0234375
[INFO] 20 : note_on channel=9 note=44 velocity=65 time=0.010937500000000001
[INFO] 21 : control_change channel=9 control=4 value=64 time=0.0125
[INFO] 22 : note_on channel=9 note=38 velocity=48 time=0.0015625
[INFO] 23 : control_change channel=9 control=4 value=90 time=0.0171875
[INFO] 24 : control_change channel=9 control=4 value=88 time=0.046875
[INFO] 25 : note_on channel=9 note=44 velocity=0 time=0.0234375
[INFO] 26 : note_on channel=9 note=38 velocity=0 time=0.0125
[INFO] 27 : control_change channel=9 control=4 value=85 time=0.010937500000000001
[INFO] 28 : control_change channel=9 control=4 value=83 time=0.046875
[INFO] 29 : note_on channel=9 note=38 velocity=47 time=0.0390625
```

- de-tokenized midi:
```
[INFO] 0 : program_change channel=0 program=0 time=0
[INFO] 1 : note_on channel=0 note=38 velocity=7 time=2.1388706249999996
[INFO] 2 : note_on channel=0 note=38 velocity=11 time=0
[INFO] 3 : note_off channel=0 note=38 velocity=7 time=0.09299437499999999
[INFO] 4 : note_off channel=0 note=38 velocity=11 time=0
[INFO] 5 : note_on channel=0 note=38 velocity=55 time=0.18598874999999998
[INFO] 6 : note_off channel=0 note=38 velocity=55 time=0.09299437499999999
[INFO] 7 : note_on channel=0 note=44 velocity=67 time=0.09299437499999999
[INFO] 8 : note_on channel=0 note=38 velocity=47 time=0
[INFO] 9 : note_off channel=0 note=44 velocity=67 time=0.09299437499999999
[INFO] 10 : note_off channel=0 note=38 velocity=47 time=0
[INFO] 11 : note_on channel=0 note=38 velocity=47 time=0.09299437499999999
[INFO] 12 : note_off channel=0 note=38 velocity=47 time=0.09299437499999999
[INFO] 13 : note_on channel=0 note=36 velocity=43 time=0.09299437499999999
[INFO] 14 : note_on channel=0 note=46 velocity=47 time=0
[INFO] 15 : note_off channel=0 note=36 velocity=43 time=0.09299437499999999
[INFO] 16 : note_off channel=0 note=46 velocity=47 time=0
[INFO] 17 : note_on channel=0 note=42 velocity=27 time=0.09299437499999999
[INFO] 18 : note_off channel=0 note=42 velocity=27 time=0.09299437499999999
[INFO] 19 : note_on channel=0 note=36 velocity=43 time=0.09299437499999999
[INFO] 20 : note_on channel=0 note=42 velocity=55 time=0
[INFO] 21 : note_off channel=0 note=36 velocity=43 time=0.09299437499999999
[INFO] 22 : note_off channel=0 note=42 velocity=55 time=0
[INFO] 23 : note_on channel=0 note=42 velocity=23 time=0.09299437499999999
[INFO] 24 : note_off channel=0 note=42 velocity=23 time=0.09299437499999999
[INFO] 25 : note_on channel=0 note=42 velocity=43 time=0.09299437499999999
[INFO] 26 : note_on channel=0 note=38 velocity=55 time=0
[INFO] 27 : note_off channel=0 note=42 velocity=43 time=0.09299437499999999
[INFO] 28 : note_off channel=0 note=38 velocity=55 time=0
[INFO] 29 : note_on channel=0 note=42 velocity=39 time=0.09299437499999999
```

- fixed deconizie
```
__________ content of detokinized __________
[INFO] 0 : program_change channel=9 program=0 time=0
[INFO] 1 : note_on channel=9 note=38 velocity=7 time=2.1388706249999996
[INFO] 2 : note_on channel=9 note=38 velocity=11 time=0
[INFO] 3 : note_off channel=9 note=38 velocity=7 time=0.09299437499999999
[INFO] 4 : note_off channel=9 note=38 velocity=11 time=0
[INFO] 5 : note_on channel=9 note=38 velocity=55 time=0.18598874999999998
[INFO] 6 : note_off channel=9 note=38 velocity=55 time=0.09299437499999999
[INFO] 7 : note_on channel=9 note=44 velocity=67 time=0.09299437499999999
[INFO] 8 : note_on channel=9 note=38 velocity=47 time=0
[INFO] 9 : note_off channel=9 note=44 velocity=67 time=0.09299437499999999
[INFO] 10 : note_off channel=9 note=38 velocity=47 time=0
[INFO] 11 : note_on channel=9 note=38 velocity=47 time=0.09299437499999999
[INFO] 12 : note_off channel=9 note=38 velocity=47 time=0.09299437499999999
[INFO] 13 : note_on channel=9 note=36 velocity=43 time=0.09299437499999999
[INFO] 14 : note_on channel=9 note=46 velocity=47 time=0
[INFO] 15 : note_off channel=9 note=36 velocity=43 time=0.09299437499999999
[INFO] 16 : note_off channel=9 note=46 velocity=47 time=0
[INFO] 17 : note_on channel=9 note=42 velocity=27 time=0.09299437499999999
[INFO] 18 : note_off channel=9 note=42 velocity=27 time=0.09299437499999999
[INFO] 19 : note_on channel=9 note=36 velocity=43 time=0.09299437499999999
[INFO] 20 : note_on channel=9 note=42 velocity=55 time=0
[INFO] 21 : note_off channel=9 note=36 velocity=43 time=0.09299437499999999
[INFO] 22 : note_off channel=9 note=42 velocity=55 time=0
[INFO] 23 : note_on channel=9 note=42 velocity=23 time=0.09299437499999999
[INFO] 24 : note_off channel=9 note=42 velocity=23 time=0.09299437499999999
[INFO] 25 : note_on channel=9 note=42 velocity=43 time=0.09299437499999999
[INFO] 26 : note_on channel=9 note=38 velocity=55 time=0
[INFO] 27 : note_off channel=9 note=42 velocity=43 time=0.09299437499999999
[INFO] 28 : note_off channel=9 note=38 velocity=55 time=0
[INFO] 29 : note_on channel=9 note=42 velocity=39 time=0.09299437499999999
```