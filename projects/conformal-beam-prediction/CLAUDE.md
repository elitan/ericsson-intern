- always use `uv` for package management. never pip.
- DeepMIMO scenarios must be in `deepmimo_scenarios/` — pass this as `scenario_folder` parameter.
- run simulation: `uv run python run_simulation.py`
- run temporal pipeline: `uv run python run_temporal.py`
- python source lives in src/beampred/
- use underscore_case for python files
- report is in report.tex (LaTeX, IEEE format)

## DeepMIMO scenario data

Scenarios live in `deepmimo_scenarios/` (git-ignored). Download locally once, reuse across runs.
Upload to cloud instances via rsync (remove `--exclude 'deepmimo_scenarios'` from rsync command).

```bash
uv run python -c "import deepmimo; deepmimo.download('boston5g_28')"
```

deepmimo.net has a daily download quota (~2GB). Small scenarios (boston5g_28 = 35MB) fit easily.
Large ones (o1_28 = 2.3GB) may need to wait for quota reset.

**IMPORTANT**: DeepMIMO v4 uses `import deepmimo` (lowercase). Do NOT use v3 API (`import DeepMIMO`).

## Cloud GPU (Vast.ai)

API key in `.env` (git-ignored). Copy `.env.example` → `.env`.

```bash
# setup (one-time)
pip install vastai
vastai set api-key $(grep VAST_API_KEY .env | cut -d= -f2)

# find cheap GPU
vastai search offers 'gpu_ram>=12 num_gpus=1 dph<0.40 inet_down>100 reliability>0.95' -o 'dph'

# create instance (pick ID from search results)
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime --disk 50 --ssh --direct

# attach SSH key + wait for status=running
vastai attach ssh <INSTANCE_ID> "$(cat ~/.ssh/id_rsa.pub)"
vastai show instance <INSTANCE_ID>    # repeat until running

# get SSH command (use ssh-url, NOT the ssh_host column)
vastai ssh-url <INSTANCE_ID>

# upload project
rsync -avz --exclude '.venv' --exclude 'data' --exclude 'figures' --exclude '__pycache__' --exclude '.env' --exclude '*.egg-info' \
  -e 'ssh -p <PORT>' . root@<HOST>:/workspace/conformal-beam-prediction/

# run on GPU
ssh -p <PORT> root@<HOST> 'cd /workspace/conformal-beam-prediction && bash run_cloud.sh'

# download results
rsync -avz -e 'ssh -p <PORT>' root@<HOST>:/workspace/conformal-beam-prediction/figures/ ./figures/

# destroy when done (billed per minute!)
vastai destroy instance <INSTANCE_ID>
```

Notes:
- `uv sync` re-downloads PyTorch even if Docker image has it. First run takes ~5min for deps.
- Subsequent runs are fast (venv cached).
- Always destroy instance when done.
