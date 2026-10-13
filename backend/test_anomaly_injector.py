import os
import sys
import json
import time
import psutil
import hashlib
import joblib
from datetime import datetime
from unittest.mock import patch

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from healing.self_heal import (
    self_heal,
    restart_monitoring_service,
    verify_monitoring_service_health,
    map_rca_to_remediation_action,
    ACTION_RESTART_COLLECTOR,
    ACTION_SAFE_FALLBACK,
    REMEDIATION_WHITELIST,
    LOG_FILE
)
from app import app, VMS

def run_anomaly_injector_tests():
    client = app.test_client()
    results = {}

    print("======================================================================")
    print("ANOMALY INJECTOR & FULL STATE SYNCHRONIZATION TEST SUITE")
    print("======================================================================\n")

    # Initial Model Hash Check
    model_path = os.path.join(BACKEND_DIR, 'models', 'trained_model.pkl')
    with open(model_path, 'rb') as f:
        initial_hash = hashlib.sha256(f.read()).hexdigest()
    expected_hash = "261e9110be2c3b45180d5210d25b4a5dd89ca7b7201727d2295273628deeb06c"
    assert initial_hash == expected_hash, f"Model hash modified! {initial_hash}"

    # ----------------------------------------------------------------------
    # TEST 1: 12 CLOUD INSTANCES & DYNAMIC HEALTH COUNT
    # ----------------------------------------------------------------------
    print("--- [TEST 1] 12 Cloud Instances & Dynamic Health Count ---")
    res_inst = client.get('/api/v1/instances')
    assert res_inst.status_code == 200
    vms_data = res_inst.get_json()
    assert isinstance(vms_data, list)
    assert len(vms_data) == 12, f"Expected 12 VMs, found {len(vms_data)}"
    healthy = sum(1 for v in vms_data if v['status'] == 'Running')
    issues = sum(1 for v in vms_data if v['status'] != 'Running')
    print(f"  Total VMs: {len(vms_data)}, Healthy: {healthy}, Issues: {issues}")
    results['1. 12 Dynamic Cloud Instances'] = "PASS"
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # TEST 2: CPU SPIKE INJECTION & RCA
    # ----------------------------------------------------------------------
    print("--- [TEST 2] CPU Spike Injection -> Real RF -> Gemini RCA -> Self-Heal ---")
    res_inject_cpu = client.post('/api/v1/inject', json={'type': 'CPU'})
    assert res_inject_cpu.status_code == 200
    d_inj_cpu = res_inject_cpu.get_json()
    assert d_inj_cpu['status'] == 'success'
    assert d_inj_cpu['fault_type'] == 'CPU_SPIKE'
    assert d_inj_cpu['metrics']['cpu'] >= 90.0
    print(f"  Injected CPU Metrics: {d_inj_cpu['metrics']}")

    # Check metrics/current reflects the injected anomaly
    res_cur = client.get('/api/v1/metrics/current')
    assert res_cur.status_code == 200
    d_cur = res_cur.get_json()
    assert d_cur['metrics']['cpu'] >= 90.0
    assert d_cur.get('injected') is True

    # Real Random Forest prediction
    with patch('app.generate_rca', return_value={'status': 'success', 'rca': 'High CPU Contention: CPU utilization surged to 98.5% causing thread starvation.'}) as mock_rca, \
         patch('app.self_heal', return_value=['Restarted Monitoring Service']) as mock_heal:
        res_pred = client.post('/api/v1/predict', json=d_cur['metrics'])
        assert res_pred.status_code == 200
        d_pred = res_pred.get_json()
        assert d_pred['is_anomaly'] is True
        assert d_pred['prediction_label'] == 'Anomaly'
        assert d_pred['anomaly_probability'] >= 0.90
        assert d_pred['overall_cloud_risk'] >= 70.0
        assert d_pred['healing_triggered'] is True
        assert d_pred['healing_status'] == 'executed'
        assert d_pred['rca'] is not None
        assert "CPU" in d_pred['rca']
        assert mock_rca.call_count == 1
        assert mock_heal.call_count == 1

    results['2. CPU Spike Injection & RCA'] = "PASS"
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # TEST 3: MEMORY LEAK INJECTION & RCA
    # ----------------------------------------------------------------------
    print("--- [TEST 3] Memory Leak Injection -> Real RF Prediction ---")
    res_inject_mem = client.post('/api/v1/inject', json={'type': 'MEMORY'})
    assert res_inject_mem.status_code == 200
    d_inj_mem = res_inject_mem.get_json()
    assert d_inj_mem['fault_type'] == 'MEMORY_LEAK'
    assert d_inj_mem['metrics']['memory'] >= 90.0

    with patch('app.generate_rca', return_value={'status': 'success', 'rca': 'Memory Saturation & Heap Leak: Memory consumption peaked at 96.5%.'}), \
         patch('app.self_heal', return_value=['Healed']):
        res_pred_mem = client.post('/api/v1/predict', json=d_inj_mem['metrics'])
        d_pred_mem = res_pred_mem.get_json()
        assert d_pred_mem['is_anomaly'] is True
        assert d_pred_mem['prediction_label'] == 'Anomaly'
        assert d_pred_mem['overall_cloud_risk'] >= 70.0
        assert "Memory" in d_pred_mem['rca']

    results['3. Memory Leak Injection & RCA'] = "PASS"
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # TEST 4: NETWORK LATENCY INJECTION & RCA
    # ----------------------------------------------------------------------
    print("--- [TEST 4] Network Latency Injection -> RF Prediction ---")
    res_inject_lat = client.post('/api/v1/inject', json={'type': 'LATENCY'})
    assert res_inject_lat.status_code == 200
    d_inj_lat = res_inject_lat.get_json()
    assert d_inj_lat['fault_type'] == 'NETWORK_LATENCY'
    assert d_inj_lat['metrics']['latency'] >= 400.0

    with patch('app.generate_rca', return_value={'status': 'success', 'rca': 'Network Latency Degradation: Latency surged to 410ms.'}), \
         patch('app.self_heal', return_value=['Healed']):
        res_pred_lat = client.post('/api/v1/predict', json=d_inj_lat['metrics'])
        d_pred_lat = res_pred_lat.get_json()
        assert d_pred_lat['is_anomaly'] is True
        assert "Latency" in d_pred_lat['rca']

    results['4. Network Latency Injection & RCA'] = "PASS"
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # TEST 5: CRITICAL MODULE CRASH INJECTION & RCA
    # ----------------------------------------------------------------------
    print("--- [TEST 5] Critical Module Crash Injection -> RF Prediction ---")
    res_inject_crash = client.post('/api/v1/inject', json={'type': 'CRASH'})
    assert res_inject_crash.status_code == 200
    d_inj_crash = res_inject_crash.get_json()
    assert d_inj_crash['fault_type'] == 'CRITICAL_CRASH'
    assert d_inj_crash['metrics']['cpu'] >= 99.0
    assert d_inj_crash['metrics']['memory'] >= 98.0

    with patch('app.generate_rca', return_value={'status': 'success', 'rca': 'Critical Node Outage & Module Crash: Process thrashing on worker node.'}), \
         patch('app.self_heal', return_value=['Healed']):
        res_pred_crash = client.post('/api/v1/predict', json=d_inj_crash['metrics'])
        d_pred_crash = res_pred_crash.get_json()
        assert d_pred_crash['is_anomaly'] is True
        assert "Crash" in d_pred_crash['rca'] or "Outage" in d_pred_crash['rca'] or "Node" in d_pred_crash['rca']

    results['5. Critical Crash Injection & RCA'] = "PASS"
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # TEST 6: RESET / AUTO-HEAL CLUSTER (RECOVERY & OVERALL CLOUD RISK)
    # ----------------------------------------------------------------------
    print("--- [TEST 6] Reset / Auto-Heal Cluster -> State Restoration ---")
    with patch('app.self_heal', return_value=['Cluster Reset & Monitoring Restarted']):
        res_heal = client.post('/api/v1/heal', json={'status': 'Anomaly'})
        assert res_heal.status_code == 200
        d_heal = res_heal.get_json()
        assert d_heal['status'] == 'success'
        assert 'restored_metrics' in d_heal
        assert d_heal['restored_metrics']['cpu'] <= 30.0

    # Prediction on normal restored metrics
    res_norm = client.post('/api/v1/predict', json=d_heal['restored_metrics'])
    d_norm = res_norm.get_json()
    assert d_norm['is_anomaly'] is False
    assert d_norm['prediction_label'] == 'Normal'
    assert d_norm['healing_triggered'] is False
    assert d_norm['healing_status'] == 'nominal'
    assert d_norm['overall_cloud_risk'] < 40.0, f"Expected risk < 40, got {d_norm['overall_cloud_risk']}"

    # Prediction analysis endpoint risk check
    res_an = client.get('/api/v1/prediction/analysis?cpu=22.0&memory=38.0&is_anomaly=false')
    assert res_an.status_code == 200
    d_an = res_an.get_json()
    assert 'overall_cloud_risk' in d_an
    assert isinstance(d_an['overall_cloud_risk'], (int, float))
    assert not (d_an['overall_cloud_risk'] is None)

    results['6. Cluster Reset & Risk Score Validity'] = "PASS"
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # TEST 7: ANTI-REPEAT DUPLICATE HEALING PROTECTION
    # ----------------------------------------------------------------------
    print("--- [TEST 7] Anti-Repeat Duplicate Healing Protection ---")
    # First anomaly triggers healing
    with patch('app.generate_rca', return_value={'status': 'success', 'rca': 'Spike'}), \
         patch('app.self_heal', return_value=['Healed']) as mock_h1:
        r1 = client.post('/api/v1/predict', json={'cpu': 98.0, 'memory': 95.0, 'disk': 88.0, 'Service Status': 1, 'Health Check': 1})
        assert r1.get_json()['healing_triggered'] is True
        assert mock_h1.call_count == 1

    # Second poll while anomaly is ongoing does NOT retrigger healing
    with patch('app.generate_rca') as mock_rca2, \
         patch('app.self_heal') as mock_h2:
        r2 = client.post('/api/v1/predict', json={'cpu': 98.0, 'memory': 95.0, 'disk': 88.0, 'Service Status': 1, 'Health Check': 1})
        assert r2.get_json()['healing_triggered'] is False
        assert r2.get_json()['healing_status'] == 'active_remediation_in_progress'
        assert mock_rca2.call_count == 0
        assert mock_h2.call_count == 0

    results['7. Anti-Repeat Protection'] = "PASS"
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # TEST 8: NO ARBITRARY SHELL EXECUTION (MALICIOUS RCA SECURITY)
    # ----------------------------------------------------------------------
    print("--- [TEST 8] Security Barrier (Zero Arbitrary Command Execution) ---")
    malicious_payload = "run rm -rf / ; cat /etc/shadow"
    action, desc = map_rca_to_remediation_action("Anomaly", rca=malicious_payload)
    assert action == ACTION_SAFE_FALLBACK, "Must map malicious payload to safe fallback!"
    assert action in REMEDIATION_WHITELIST, "Must be in whitelist!"

    results['8. Security Whitelist Barrier'] = "PASS"
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # TEST 9: FRONTEND SYNCHRONIZATION & ANOMALY INJECTOR VISIBILITY
    # ----------------------------------------------------------------------
    print("--- [TEST 9] Frontend Sync & Anomaly Injector Visibility ---")
    f_major = os.path.join(BACKEND_DIR, '..', 'frontend', 'index.html')
    f_root = os.path.join(os.path.dirname(os.path.dirname(BACKEND_DIR)), 'frontend', 'index.html')
    
    with open(f_major, 'r', encoding='utf-8') as f:
        html_major = f.read()
    if os.path.exists(f_root):
        with open(f_root, 'r', encoding='utf-8') as f:
            html_root = f.read()
        assert html_major == html_root, "Frontend files are out of sync!"
    assert "triggerAnomalyInjection" in html_major
    assert "Inject Fault" in html_major
    assert "triggerAutoHeal" in html_major

    results['9. Frontend Sync & Anomaly Injector Panels'] = "PASS"
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # TEST 10: MODEL PROTECTION & SHA-256 CHECK
    # ----------------------------------------------------------------------
    print("--- [TEST 10] Model SHA-256 Check ---")
    with open(model_path, 'rb') as f:
        final_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"  Initial SHA-256: {initial_hash}")
    print(f"  Final SHA-256:   {final_hash}")
    assert initial_hash == final_hash == expected_hash, "MODEL HASH HAS CHANGED!"
    results['10. Model Protection (100% Intact)'] = "PASS"
    print("  Status: PASS\n")

    # Clean reset
    client.post('/api/v1/heal', json={'status': 'Normal'})

    print("======================================================================")
    all_pass = all(v == "PASS" for v in results.values())
    print(f"ANOMALY INJECTOR TEST SUITE STATUS: {'ALL 10 TESTS PASSED (100%)' if all_pass else 'FAILED'}")
    print("======================================================================")
    return all_pass

if __name__ == '__main__':
    ok = run_anomaly_injector_tests()
    sys.exit(0 if ok else 1)
