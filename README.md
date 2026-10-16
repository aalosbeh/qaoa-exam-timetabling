# qaoa-exam-timetabling

Quality-adaptive warm-start QAOA for university exam timetabling, evaluated on an
ideal simulator, a calibrated noisy simulator and IBM Quantum hardware
(Qiskit 2.x, qiskit-ibm-runtime 0.50).

The exam timetabling problem is encoded as weighted graph colouring in a one-hot
QUBO (`src/qubo.py`), solved with QAOA whose initial state and mixer are biased
towards a classical greedy timetable with a regularisation that adapts to the
greedy solution's quality (`src/warm_start.py`, `src/circuits.py`), and
benchmarked against Simulated Annealing and exact enumeration.

## Quick start

```bash
git clone https://github.com/aalosbeh/qaoa-exam-timetabling.git
cd qaoa-exam-timetabling
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                            # 18 tests, about 3 s
python run_simulator.py --courses 6 --reps 2         # one run, prints a timetable
```

## Scripts

| Script | Purpose |
| --- | --- |
| `run_simulator.py` | One end-to-end run on the ideal simulator with all five initialisations; prints the timetable |
| `run_experiments.py` | Experiment grid E1/E2/E3/E6 (depth, initialisation, size, penalty) with seeds; resumable; writes `results/<tag>_runs.csv` |
| `run_hardware.py` | E4/E5: evaluates stored parameters on the ideal simulator, a noisy simulator (`FakeFez` by default) and, with `--real`, an IBM QPU, raw and readout-mitigated |
| `analyse.py` | Figures (PDF/PNG), Wilcoxon tests and LaTeX tables from the results |
| `scripts/make_synthetic.py` | Random weighted conflict graphs with a spread of greedy quality |

Typical sequence:

```bash
python run_experiments.py --sizes 4,5,6,8 --reps 1,2,3 --lambdas 2 --seeds 20 --tag full
python run_experiments.py --sizes 6 --reps 2 --lambdas 0.5,1,4,8 --seeds 20 --methods ramp,ws_adaptive --tag full
python run_hardware.py --tag full --courses 6 --reps 1,2            # noisy simulator
python run_hardware.py --tag full --courses 6 --reps 1,2 --real     # + real QPU (needs an account)
python analyse.py --tag full --out figures
```

## IBM Quantum access

Create a free account on the IBM Quantum Platform (Open Plan: 10 minutes of QPU
time per 28 days), copy the API key and save it once:

```bash
python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; \
S.save_account(channel='ibm_quantum_platform', token='PASTE_API_KEY', overwrite=True)"
```

The classic `ibm_quantum` channel was retired on 1 July 2025 and no longer works.
`run_hardware.py --real` picks the least busy operational QPU with enough qubits
and submits eight jobs for two depths (about three minutes of QPU time).

## Data format

`data/exams.csv` holds one row per pair of courses with at least one student in
common: `course_a,course_b,shared_students`. The shipped file is a synthetic
development instance of eight courses; the anonymised departmental instance
replaces it at week 3 of the project (see `data/DATA_CARD.md`).

## Layout

```
src/qubo.py        conflict graph, QUBO, Ising mapping, decoding, exact references
src/classical.py   weighted greedy timetable, Simulated Annealing (QUBO and native moves)
src/warm_start.py  quality score q, eps(q), biased initial state, the five initialisations
src/circuits.py    explicit QAOA / warm-start QAOA circuits
src/qaoa_core.py   samplers, hybrid loop (COBYLA), metrics
src/mitigation.py  M3-style readout mitigation for counts
tests/             unit tests (QUBO, Ising mapping, circuits, warm start)
```

## Licence

Code: MIT. Data card and dataset: CC BY 4.0.
