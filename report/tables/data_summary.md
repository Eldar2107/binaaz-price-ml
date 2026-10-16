### Data summary (cleaning log, split sizes, tier threshold)

```
--- cleaning log ---
de-duplicate by listing url (keep latest scrape)  100775 ->   64454  (removed 36321)
drop rows with unparseable area/price              64454 ->   64454  (removed 0)
price outside [10,000, 5,000,000] AZN              64454 ->   63997  (removed 457)
area outside [10, 20,000] m2                       63997 ->   63909  (removed 88)
coordinates outside Azerbaijan                     63909 ->   63908  (removed 1)
final shape: (63908, 15)
```

- features after encoding: 164
- split sizes train / val / test: 44,732 / 9,588 / 9,588
- tier threshold (median of TRAIN prices): 210,000 AZN
- premium share train / val / test: 0.494 / 0.494 / 0.494
