import numpy as np
import pytest
from pydantic import ValidationError
from backend.app.realtime import LockMachine, compute_metrics, merge_intervals
from backend.app.schemas import E3, Interval, Service
from backend.app.store import hash_password, verify_password
from backend.app.models import unit


def test_password_salted_and_verified():
    a,b=hash_password('secret123'),hash_password('secret123')
    assert a!=b and 'secret123' not in a
    assert verify_password('secret123',a)
    assert not verify_password('wrong',a)


def test_state_machine_hysteresis_and_silence():
    m=LockMachine(.7,.5,2)
    assert m.advance(.8)==('CANDIDATE',None)
    assert m.advance(.4)==('NON_TARGET',None)
    assert m.advance(.8)==('CANDIDATE',None)
    assert m.advance(.8)==('LOCKED','LOCK')
    assert m.advance(.6)==('LOCKED',None)
    assert m.advance(None)==('UNLOCKED','UNLOCK')
    assert m.advance(.8)==('CANDIDATE',None)
    assert m.advance(.8)==('LOCKED','LOCK')


def test_ground_truth_metrics_no_backdating():
    rows=[{'decision_ms':1500,'state':'CANDIDATE','event':None},
          {'decision_ms':2000,'state':'LOCKED','event':'LOCK'},
          {'decision_ms':4500,'state':'UNLOCKED','event':'UNLOCK'},
          {'decision_ms':7000,'state':'LOCKED','event':'LOCK'},
          {'decision_ms':8500,'state':'UNLOCKED','event':'UNLOCK'}]
    assert compute_metrics(rows,10000)['false_locks'] is None
    m=compute_metrics(rows,10000,[{'start_ms':1000,'end_ms':4000},{'start_ms':6000,'end_ms':8000}])
    assert m['first_lock_latency_ms']==1000
    assert m['unlock_latency_ms']==500
    assert m['relock_latency_ms']==1000
    assert m['false_locks']==0
    assert m['target_coverage']==.6
    no_target=compute_metrics(rows,10000,[])
    assert no_target['false_locks']==2 and no_target['target_coverage'] is None


def test_missed_target_and_overlapping_annotation():
    rows=[{'decision_ms':2000,'state':'NON_TARGET','event':None}]
    m=compute_metrics(rows,4000,[{'start_ms':0,'end_ms':3000}])
    assert m['first_lock_latency_ms'] is None and m['missed_target_intervals']==1 and m['target_coverage']==0
    assert merge_intervals([{'start_ms':0,'end_ms':3},{'start_ms':2,'end_ms':5}],10)==[{'start_ms':0.,'end_ms':5.}]
    with pytest.raises(ValueError): merge_intervals([{'start_ms':0,'end_ms':20}],10)


@pytest.mark.parametrize('data',[{'profile_id':'p','lock_threshold':.4,'unlock_threshold':.6},{'profile_id':'p','confirm_windows':0},{'profile_id':'p','lock_threshold':float('nan')}])
def test_invalid_realtime_config(data):
    with pytest.raises(ValidationError): E3(**data)


def test_invalid_intervals_and_service():
    with pytest.raises(ValidationError): Interval(start_ms=100,end_ms=50)
    with pytest.raises(ValidationError): Service(name='a',base_url='file:///secret',models=['x'],default_model='x')
    with pytest.raises(ValidationError): Service(name='a',base_url='http://localhost/v1',models=['x'],default_model='y')


def test_invalid_embedding():
    with pytest.raises(ValueError): unit(np.zeros(256))
    with pytest.raises(ValueError): unit(np.full(256,np.nan))
    assert np.isclose(np.linalg.norm(unit(np.ones(256))),1)
