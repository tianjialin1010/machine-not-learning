"""Exercise every real industrial model and save a compact inference audit."""
import argparse
import json
import math
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request(base, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(base + path, data=body, headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=90) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def verify(base):
    status, catalog = request(base, '/v1/industrial/catalog')
    assert status == 200
    identifiers = {'ai4i', 'skab', 'steel', 'secom', 'tep'}
    assert {item['id'] for item in catalog['scenarios']} == identifiers
    report = {'ok': True, 'base': base, 'scenarios': {}, 'error_checks': {}}
    sample_ids = {}
    for item in catalog['scenarios']:
        scenario = item['id']
        assert item['available'], item
        assert item['model']['device'].startswith('cuda'), item['model']
        status, samples = request(base, '/v1/industrial/samples?scenario=' + scenario)
        assert status == 200 and samples['scenario_id'] == scenario
        assert len(samples['samples']) >= 2
        runs = []
        for sample in samples['samples'][:2]:
            assert sample['split'] == 'test'
            payload = {'scenario_id': scenario, 'sample_id': sample['id']}
            status, result = request(base, '/v1/industrial/decide', payload)
            assert status == 200, result
            assert result['scenario_id'] == scenario and result['sample_id'] == sample['id']
            assert result['mode'] == 'live_model' and result['source'] == 'heldout_dataset'
            assert result['model']['sha256'] == item['model']['sha256']
            assert result['model']['device'].startswith('cuda')
            assert result['sample']['features'] == sample['features']
            assert result['display_features'] == sample['display_features']
            prediction = result['prediction']
            probabilities = prediction['probabilities']
            assert set(probabilities) == set(item['classes'])
            assert prediction['label'] in probabilities
            assert all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1 for value in probabilities.values())
            assert math.isclose(sum(probabilities.values()), 1, abs_tol=.0001)
            if scenario == 'steel':
                assert prediction['is_anomaly'] is None and prediction['fault_probability'] is None
            else:
                assert type(prediction['is_anomaly']) is bool and 0 <= prediction['fault_probability'] <= 1
            runs.append({'sample_id': sample['id'], 'expected_class': sample['expected_class'],
                         'prediction': prediction, 'trace_id': result['trace_id'], 'latency_ms': result['latency_ms']})
        status, repeat = request(base, '/v1/industrial/decide', {'scenario_id': scenario, 'sample_id': samples['samples'][0]['id']})
        assert status == 200 and repeat['prediction'] == runs[0]['prediction']
        assert repeat['trace_id'] != runs[0]['trace_id']
        sample_ids[scenario] = samples['samples'][0]['id']
        report['scenarios'][scenario] = {'model': item['model'], 'feature_count': item['feature_count'], 'runs': runs, 'repeat_is_deterministic': True}
    for name, payload, expected in (
        ('unknown_scenario', {'scenario_id': 'unknown', 'sample_id': 'x'}, 422),
        ('unknown_sample', {'scenario_id': 'skab', 'sample_id': 'not-a-sample'}, 404),
        ('cross_scenario_sample', {'scenario_id': 'steel', 'sample_id': sample_ids['skab']}, 404),
        ('unexpected_features', {'scenario_id': 'skab', 'sample_id': sample_ids['skab'], 'expected_class': 'normal'}, 422),
    ):
        status, body = request(base, '/v1/industrial/decide', payload)
        assert status == expected and body.get('mode') != 'live_model', (name, status, body)
        report['error_checks'][name] = status
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:19000')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.base.rstrip('/'))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'ok': result['ok'], 'scenarios': list(result['scenarios']), 'error_checks': result['error_checks']}, ensure_ascii=False))
