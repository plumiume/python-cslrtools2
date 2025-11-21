# Missing SHA256 hashes for Windows wheels in torchvision 0.24.1 (CUDA 12.8) on download.pytorch.org

## Summary  
The official PyTorch wheel index (<https://download.pytorch.org/whl/cu128/>) is missing SHA256 hash fragments for Windows wheels of **torchvision 0.24.1** in its [PEP 503 Simple API](https://peps.python.org/pep-0503/) listing.

This breaks reproducible installations on Windows when using hash-validating package managers (such as [`uv`](https://github.com/astral-sh/uv)), which reject these wheels due to hash mismatches.

---

## Details  

**Observed behavior:**
- Linux wheels (manylinux) **include** the `#sha256=` fragment:  
  ```
  torchvision-0.24.1%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=cf84eae1d2d12a7d261a7496eca00dd927b71792011b1e84d4162c950eb3201d
  ```
- Windows wheels **omit** it:  
  ```
  torchvision-0.24.1%2Bcu128-cp312-cp312-win_amd64.whl
  ```
  (There is a `data-dist-info-metadata` attribute with a hash, but no `#sha256=` fragment in the href.)

**Comparison:**  
In torchvision 0.24.0, Windows wheels included the expected fragment:  
```
torchvision-0.24.0%2Bcu128-cp312-cp312-win_amd64.whl#sha256=1aa36ac00106e1381c38348611a1ec0eebe942570ebaf0490f026b061dfc212c
```

**Expected:**  
All wheels in the Simple API should include the `#sha256=` fragment, per [PEP 503](https://peps.python.org/pep-0503/).

According to PEP 503 specification:
> A repository MAY include a hash digest of the file as a URL fragment in the form `#<hashname>=<hashvalue>`, where `<hashname>` is the name of the hash function (such as `sha256`, `sha384`, `sha512`) and `<hashvalue>` is the hex-encoded digest. Multiple hashes may be included by separating them with `&`.

While PEP 503 states this is optional ("MAY"), it is strongly recommended for security and reproducibility. When hashes are provided for some platforms but missing for others in the same version, it creates inconsistency and breaks tools that rely on hash verification for reproducible builds.

**Actual:**  
Windows wheels for torchvision 0.24.1+cu128 are missing the `#sha256` fragment, while Linux wheels are correct.

---

## Impact  

Tools that verify package hashes (such as `uv`) fail with an error similar to:

```
Hash mismatch for https://download.pytorch.org/whl/cu128/torchvision-0.24.1+cu128-cp312-cp312-win_amd64.whl
```

This prevents reproducible or locked installations on Windows.

---

## Environment  

**System:**
- OS: Windows  
- Shell: PowerShell (`pwsh.exe`)

**Tool Versions:**
- uv: 0.9.7 (`0adb44480`, built 2025-10-30)  
- Python: ≥ 3.12  

**Project setup (pyproject.toml):**
- Custom index: `https://download.pytorch.org/whl/cu128`  
- Index strategy: `unsafe-best-match`  
- Dependencies:
  - `torch >= 2.9.0`
  - `torchvision` (no version constraint initially)

**Command:**
```
uv sync --group mediapipe --index-strategy unsafe-best-match
```

**Error:**
```
Failed to download torchvision==0.24.1+cu128
Hash mismatch for torchvision==0.24.1+cu128
```

**Expected (from lock file):**
```
sha256:cf84eae1d2d12a7d261a7496eca00dd927b71792011b1e84d4162c950eb3201d
sha256:5f7c5e0fa08d2cbee93b6e04bbedd59b5e11462cff6cefd07949217265df2370
sha256:5ae2dc0f582215b078d7fd52410fe51f79b801770c53e7cfb8ad04316283017d
sha256:3b72e32377e5e91398ddc4579c77784b269652a5795f4b20a5a1d4c80e9bd3dd
sha256:45792b58c2a9761da4e1d9d12c4bf5140b6250ef9210f42f716f284cff5566ea
```

**Computed (actual downloaded file):**
```
sha256:33ecea57afa1daeedfed443a8a0cb8e4b0b403fdf18c2a328ba6f9069d403384
```

---

## Investigation  

Inspecting the HTML at <https://download.pytorch.org/whl/cu128/torchvision/> shows:

| Version | Platform | Example href | SHA256 fragment present? |
|----------|-----------|---------------|---------------------------|
| 0.24.1   | Linux (manylinux) | `...#sha256=cf84...` | ✅ Yes |
| 0.24.1   | Windows (win_amd64) | `...win_amd64.whl` | ❌ No |
| 0.24.0   | Windows (win_amd64) | `...#sha256=1aa36...` | ✅ Yes |

This confirms that only Windows wheels of version 0.24.1+cu128 lack PEP 503-compliant hashes.

---

## Verification

Manual download and hash computation confirms the mismatch:

```powershell
Invoke-WebRequest -Uri "https://download.pytorch.org/whl/cu128/torchvision-0.24.1%2Bcu128-cp312-cp312-win_amd64.whl" -OutFile "temp.whl"
Get-FileHash -Path "temp.whl" -Algorithm SHA256
# Result: 33ECEA57AFA1DAEEDFED443A8A0CB8E4B0B403FDF18C2A328BA6F9069D403384
```

This matches the "Computed" hash but differs from the "Expected" Linux hashes in the lock file.

**Affected Python versions:**
- cp310, cp311, cp312, cp313, cp313t, cp314, cp314t (all Windows builds)

---

## Root Cause  

`download.pytorch.org` is not including `#sha256=` fragments in the Simple Repository API listing for **Windows** wheels of `torchvision==0.24.1+cu128`.  
Other platforms (Linux) and previous versions (e.g. 0.24.0) are unaffected.

---

## Workaround  

Pinning `torchvision==0.24.0+cu128` resolves the issue, as those wheels contain proper hashes.

---

## Request  

Please verify whether this issue is known, and if not, update the repository metadata for `torchvision 0.24.1+cu128` Windows wheels so that all entries include proper SHA256 fragments per PEP 503.  

This will restore compatibility with hash-verifying installers like `uv` and ensure consistent reproducibility across platforms.
