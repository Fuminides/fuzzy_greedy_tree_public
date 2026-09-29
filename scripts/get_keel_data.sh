#!/usr/bin/env bash
# Download the 30 KEEL benchmarks used in the paper and verify them.
# Usage: scripts/get_keel_data.sh [target_dir]   (default: ../keel_datasets)
# Then export KEEL_DIR=<target_dir> before running the experiments.
set -euo pipefail
TARGET="${1:-../keel_datasets}"
URL="https://sci2s.ugr.es/keel/dataset/data/classification"
mkdir -p "$TARGET"
fail=0
while read -r name sum; do
  [ -z "$name" ] && continue
  mkdir -p "$TARGET/$name"
  if [ ! -f "$TARGET/$name/$name.dat" ]; then
    tmp=$(mktemp -d)
    curl -sfL "$URL/$name.zip" -o "$tmp/$name.zip"
    unzip -q -o "$tmp/$name.zip" -d "$tmp"
    cp "$tmp/$name.dat" "$TARGET/$name/$name.dat"
    rm -rf "$tmp"
  fi
  # Checksum of the content with CR and trailing whitespace removed (KEEL
  # mirrors differ only in line endings).
  got=$(tr -d '\r' < "$TARGET/$name/$name.dat" | sed 's/[[:space:]]*$//' | sha256sum | cut -d' ' -f1)
  if [ "$got" = "$sum" ]; then echo "ok   $name"; else echo "FAIL $name (checksum mismatch)"; fail=1; fi
done <<'LIST'
australian ba935a764ba14bb16c065fe8faa84719cbba08ce273c21615796604900572d3a
balance 08eccbdb27e6a1dfd7143ca64f40c203b1d417399f8be54e0d73648e036911f7
banana 9684c0f72a27f56319e4196f4aa246966a87d3d84dab954cc363f033bf8f4943
bupa c4c10002c0fef0772baa2ed182b65806918608e773b27a6ec9c59f3c75dd02d3
contraceptive ed3110507599b69ed2d66a5ac387a488c254c63d698c87b26bf880ed27ca668b
crx 8df3432a85dc6428055d863b7b438a7be2307fffa794474753c1fb5de09fa7ac
ecoli 846c0ea3915fc0141025579858d90a883d0120e8a6eb16961a8c6938ae3b9307
german 116d4b8f18bdddc269968f934f67f65e2b666c7d90b4c43f388595f29ca67d96
glass 3e6a8bc0e063ccdbc83cb2e2f287363f116cfe0f676f2d23dd1cc011d0e13bd1
heart 4af6cfb7f371a1d49ab1ac2aa2f6d180a103e00b4ba4f8ae0dae826adc98f203
ionosphere dafcc9b21fc18e72384d90724a84b98baa759622d9c0a47298badf5b94b6de07
magic 0b52b2666b0c9adfb752f81d37be2e84d2b57af0ce71c1b8c120f826a0ac5d88
mammographic f82a0213037b3b4afd09bf159a5baffdd5787aeb147f2676ad67e69d18833ae9
optdigits 48409d9a43308661e2d968fda8bdd7aa70538e4079d716628e01c5187a2f3baf
penbased fcf2eacbc6d3e33622ec7c33c3755fee87caa25e9f0851ea585465afca460d78
phoneme fe7e701fc5b4a14235f84f2a45ed0f4ac071a9adf59cd1297e69bab4e395d6c6
pima 9cab0ea826bd8ec198bcf800f3d8bec326b03e63e2211547e316b5b6913d30f2
ring 8c8174d88e6ca45f22b42fc5c23d4ccd78ba52377aa57e3391d9c409832ab76a
saheart 1a88a425427acdfd24c52f83c9d04f2586a040ea6c53179ba6ebe649c0caca16
satimage 619f3afc559e145afac3efc6b2da66d74a880839511c6816b9c1b8424c4d0537
segment 5365380db361613fadc0e4f6843aed50853ed892fb91010084868fe545d4e865
spambase 1ba26d9bcd80b5f35acab05f0b6d7128bd91d8278c3dca3e2e07f35980602019
spectfheart 29370e8ed48bae7bcc06c706944fd7700c1898957766a88c580aa6ee179ab23f
texture 3227c9bed93060b9577842ddc185bb2a2b83577a7654a10bff37e400d00702d7
twonorm 1adb9c10391ff7cae6f3533c6d98394e43d9d64569ad79cc489c4872fde417af
vehicle 4eddc5ef8bed9dd64770e42f10fc2d8f2bb05eddda8c80898326061cb0577e61
vowel c1f4c5aec360c6dd3a52e4194200f11453120fb2211714828b7423650397fd1a
wdbc 72c6d8b0cc18439d6eefa43790f961f7a76b3e0102b2ce1f4b968865de040f8a
wine 010d7f9190b193a3896ed280830172625299b8c7a93cd575fc4b56d2b9b7fb9c
wisconsin dd081b5c5772a204254f500c289b20bf510aaa0b0e7f37d974531105f2d94fb1
LIST
exit $fail
