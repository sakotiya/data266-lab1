# Shared raw data — Monet / photo (Kaggle `gan-getting-started`)

300 Monet paintings (`monet_jpg/`) and 7,038 photographs (`photo_jpg/`), unpaired. ~113 MB.

Raw data is **not committed**.

## Fetch

The same 300 Monet + 7,038 photo images (256x256 JPEG) also ship in the instructor's
`Part3_export/dataset/dataset/` bundle; copy `monet_jpg/` and `photo_jpg/` from there to skip Kaggle.

Otherwise requires a Kaggle account and `~/.kaggle/kaggle.json`, and you must **join the competition
first** or the download 403s:

```bash
pip install kaggle
kaggle competitions download -c gan-getting-started -p /tmp
unzip -q /tmp/gan-getting-started.zip -d task3_gan/data
```

Expected afterwards:

```
task3_gan/data/monet_jpg/   300 files
task3_gan/data/photo_jpg/  7038 files
```

Verify: `ls task3_gan/data/monet_jpg | wc -l` -> 300

## Notes

- The 23:1 domain imbalance means "one epoch" needs defining before anyone trains -
  `epoch_definition` in the config states which domain sets the epoch length.
- The competition scores **photo → Monet** with MiFID and expects a **zip of generated
  images**, not the `submission.csv` the brief's folder tree shows. Confirm with the instructor.
