"""Check released EWCS 2024 employee-voice results and aggregate derivations.

Usage:
    python code/verify_public_results.py --out /tmp/ewcs-verification --figures

The verification uses published aggregate tables. It does not read respondent
records or estimate any survey models.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,subprocess,sys,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from check_aggregates import check
from verify_scientific_files import check_manifest


def records(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Published files remain subject to SHA-256 verification. Calculated floating-
# point results are compared numerically across supported Python environments.
FLOAT_FIELDS = {
    'response_state_comparisons.csv': {
        'estimate_pp', 'se_pp', 'lo95_pp', 'hi95_pp', 'p_pointwise',
        'covariance_no_unknown_pp2',
    },
    'voice_item_response_counts.csv': {
        'voice_incomplete_percent_within_response',
        'response_percent_within_voice_complete',
        'response_percent_within_voice_incomplete',
    },
}


def compare_derived_table(released: Path, recalculated: Path) -> None:
    float_fields = FLOAT_FIELDS[released.name]
    with released.open(encoding='utf-8-sig', newline='') as stream:
        expected_reader = csv.DictReader(stream)
        expected_fields = expected_reader.fieldnames
        expected = list(expected_reader)
    with recalculated.open(encoding='utf-8-sig', newline='') as stream:
        actual_reader = csv.DictReader(stream)
        actual_fields = actual_reader.fieldnames
        actual = list(actual_reader)
    if expected_fields != actual_fields or len(expected) != len(actual):
        raise AssertionError(f'Unexpected structure in calculated {released.name}')
    if expected_fields is None or not float_fields.issubset(expected_fields):
        raise AssertionError(f'Missing required columns in {released.name}')
    for number, (original, computed) in enumerate(zip(expected, actual), start=1):
        for field in expected_fields:
            left, right = original[field], computed[field]
            if field in float_fields:
                x, y = float(left), float(right)
                if not (math.isfinite(x) and math.isfinite(y)
                        and math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-8)):
                    raise AssertionError(
                        f'Calculated {released.name}, row {number}, {field}: '
                        f'published={left}, recalculated={right}')
            elif left != right:
                raise AssertionError(
                    f'Calculated {released.name}, row {number}, {field}: '
                    f'published={left!r}, recalculated={right!r}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--figures',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    check_manifest()
    a=check(ROOT/'results')
    with tempfile.TemporaryDirectory(prefix='ewcs-aggregate-') as d:
        temp=Path(d)
        commands=[
            [sys.executable,str(ROOT/'code/response_state_comparisons.py'),'--results',str(ROOT/'results'),'--out',str(temp)],
            [sys.executable,str(ROOT/'code/voice_item_response.py'),'--results',str(ROOT/'results'),'--out',str(temp)],
        ]
        for command in commands:subprocess.run(command,check=True,capture_output=True,text=True)
        names=('response_state_comparisons.csv','voice_item_response_counts.csv')
        for name in names:
            compare_derived_table(ROOT/'results'/name, temp/name)
        resp=records(ROOT/'results/response_state_comparisons.csv')
        voice=records(ROOT/'results/voice_item_response_counts.csv')
        assert len(resp)==7 and len(voice)==3
        assert [int(row['voice_incomplete_n']) for row in voice]==[210,74,156]
        assert sum(int(row['routed_q57_valid_n']) for row in voice)==6988
        assert sum(int(row['both_voice_items_valid_n']) for row in voice)==6548
        assert abs(float(resp[0]['estimate_pp'])+7.90977631)<1e-7
    figure_receipt=None
    if args.figures:
        from PIL import Image
        figures=args.out/'figures'
        subprocess.run([sys.executable,str(ROOT/'code/plot_figures.py'),'--tables-dir',str(ROOT/'results'),
            '--baseline-csv',str(ROOT/'results/BASELINE_Q57.csv'),'--out',str(figures)],
            capture_output=True,text=True,check=True)
        subprocess.run([sys.executable,str(ROOT/'code/apply_figure1_crop_v1.py'),'--figures-dir',str(figures)],
            capture_output=True,text=True,check=True)
        with Image.open(figures/'Figure1_Procedure_profiles.png') as img:
            size=list(img.size)
            assert size==[2868,1155],f'Unexpected Figure 1 size: {size}'
        figure_receipt={'figure_families':3,'figure_1_cropped_size_pixels':size,'source':'Released aggregate CSVs'}
    receipt={
        'status':'PASS',
        'study':'EWCS 2024 second edition, UKDS study 9511',
        'respondents_in_analytic_domain':6548,
        'primary_aggregate_table_rows':a['empirical_table_rows'],
        'inferential_records':a['inferential_records'],
        'covariance_derived_response_comparisons':7,
        'voice_item_completeness_records':3,
        'response_partition':a['response_partition'],
        'Holm_family':a['three_contrast_Holm'],
        'respondent_records_read':0,
        'source_to_model_reestimation':False,
        'matched_files':{name:digest(ROOT/'results'/name) for name in names},
        'figures':figure_receipt,
    }
    (args.out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    main()
