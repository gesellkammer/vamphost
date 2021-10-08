
import vamp
import numpy as np
import vamp.frames as fr

plugin_key = "vamp-test-plugin:vamp-test-plugin"
plugin_key_freq = "vamp-test-plugin:vamp-test-plugin-freq"

rate = 44100.0

# Throughout this file we have the assumption that the plugin gets run with a
# blocksize of 1024, and with a step of 1024 for the time-domain version or 512
# for the frequency-domain one. That is certainly expected to be the norm for a
# plugin like this that declares no preference, and the Python Vamp module is
# expected to follow the norm

blocksize = 1024
eps = 1e-6

def input_data(n):
    # start at 1, not 0 so that all elts are non-zero
    return np.arange(n) + 1    

def test_collect_runs_at_all():
    buf = input_data(blocksize * 10)
    rdict = vamp.collect(buf, rate, plugin_key, "input-timestamp")
    step, results = rdict["vector"]
    assert results != []

##!!! add test for default output
    
def test_collect_one_sample_per_step():
    buf = input_data(blocksize * 10)
    rdict = vamp.collect(buf, rate, plugin_key, "input-timestamp")
    step, results = rdict["vector"]
    assert abs(float(step) - (1024.0 / rate)) < eps
    assert len(results) == 10
    for i in range(len(results)):
        # The timestamp should be the frame number of the first frame in the
        # input buffer
        expected = i * blocksize
        actual = results[i]
        assert actual == expected

def test_process_summary_param():
    buf = input_data(blocksize * 10)
    rdict = vamp.collect(buf, rate, plugin_key, "input-summary", { "produce_output": False })
    assert ("vector" in rdict)
    step, results = rdict["vector"]
    assert len(results) == 0
    rdict = vamp.collect(buf, rate, plugin_key, "input-summary", { "produce_output": True })
    assert ("vector" in rdict)
    step, results = rdict["vector"]
    assert len(results) > 0

def test_process_summary_param_kwargs_1():
    buf = input_data(blocksize * 10)
    rdict = vamp.collect(plugin_key = plugin_key, output = "input-summary", parameters = { "produce_output": False }, data = buf, sample_rate = rate)
    assert ("vector" in rdict)
    step, results = rdict["vector"]
    assert len(results) == 0

def test_process_summary_param_kwargs_2():
    buf = input_data(blocksize * 10)
    rdict = vamp.collect(plugin_key = plugin_key, output = "input-summary", data = buf, sample_rate = rate)
    assert ("vector" in rdict)
    step, results = rdict["vector"]
    assert len(results) > 0
    
def test_process_summary_param_kwargs_3():
    buf = input_data(blocksize * 10)
    rdict = vamp.collect(plugin_key = plugin_key, output = "input-summary", data = buf, sample_rate = rate, process_timestamp_method = vamp.vampyhost.SHIFT_DATA)
    assert ("vector" in rdict)
    step, results = rdict["vector"]
    assert len(results) > 0

def test_process_summary_param_kwargs_fail():
    buf = input_data(blocksize * 10)
    try:
        rdict = vamp.collect(plugin_key = plugin_key, output = "input-summary", data = buf, sample_rate = rate, process_timestamp_method = vamp.vampyhost.SHIFT_DATA, unknown_argument = 1)
    except Exception: # unknown kwarg
        pass

def test_collect_fixed_sample_rate():
    buf = input_data(blocksize * 10)
    rdict = vamp.collect(buf, rate, plugin_key, "curve-fsr")
    step, results = rdict["vector"]
    assert abs(float(step) - 0.4) < eps
    assert len(results) == 10
    for i in range(len(results)):
        assert abs(results[i] - i * 0.1) < eps

def test_collect_fixed_sample_rate_2_vector():
    # This one has discontinuities and overlaps, so it should be
    # returned to us in two forms: as a simple vector, and as an
    # additional tracks shape with one vector for each separate
    # bit. This test covers the simple vector part
    buf = input_data(blocksize * 10)
    rdict = vamp.collect(buf, rate, plugin_key, "curve-fsr-timed")
    step, results = rdict["vector"]
    assert abs(float(step) - 0.4) < eps
    assert len(results) == 10
    for i in range(len(results)):
        assert abs(results[i] - i * 0.1) < eps

def test_collect_fixed_sample_rate_2_tracks():
    # This one has discontinuities and overlaps, so it should be
    # returned to us in two forms: as a simple vector, and as an
    # additional tracks shape with one vector for each separate
    # bit. This test covers the tracks part.
    buf = input_data(blocksize * 10)
    rdict = vamp.collect(buf, rate, plugin_key, "curve-fsr-timed")
    results = rdict["tracks"]
    assert len(results) == 8
    expected_starts = [ 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 4.0, 4.0 ]
    expected_lengths = [ 1, 1, 2, 1, 1, 2, 1, 1 ]
    expected_values = [ [ 0.0 ], [ 0.1 ], [ 0.2, 0.3 ], [ 0.4 ], [ 0.5 ],
                        [ 0.6, 0.7 ], [ 0.8 ], [ 0.9 ] ] 
    for i in range(len(results)):
        track = results[i]
        assert abs(float(track["step"]) - 0.4) < eps
        assert abs(float(track["start"]) - expected_starts[i]) < eps
        assert len(track["values"]) == expected_lengths[i]
        for j in range(expected_lengths[i]):
            assert abs(track["values"][j] - expected_values[i][j]) < eps
        
def test_collect_variable_sample_rate():
    buf = input_data(blocksize * 10)
    rdict = vamp.collect(buf, rate, plugin_key, "curve-vsr")
    results = rdict["list"]
    assert len(results) == 10
    i = 0
    for r in results:
        assert r["timestamp"] == vamp.vampyhost.RealTime('seconds', i * 0.75)
        assert abs(r["values"][0] - i * 0.1) < eps
        i = i + 1

def test_collect_grid_one_sample_per_step():
    buf = input_data(blocksize * 10)
    rdict = vamp.collect(buf, rate, plugin_key, "grid-oss")
    step, results = rdict["matrix"]
    assert abs(float(step) - (1024.0 / rate)) < eps
    assert len(results) == 10
    for i in range(len(results)):
        expected = np.array([ (j + i + 2.0) / 30.0 for j in range(0, 10) ])
        assert (abs(results[i] - expected) < eps).all()
