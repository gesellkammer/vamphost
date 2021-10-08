
import vampyhost as vh
import vamp

plugin_key = "vamp-test-plugin:vamp-test-plugin"

plugin_key_freq = "vamp-test-plugin:vamp-test-plugin-freq"

rate = 44100

expectedVersion = 4

def test_plugin_exists():
    assert plugin_key in vh.list_plugins()
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    assert "pluginVersion" in plug.info
    if plug.info["pluginVersion"] != expectedVersion:
        print("Test plugin version " + str(plug.info["pluginVersion"]) + " does not match expected version " + str(expectedVersion))
    assert plug.info["pluginVersion"] == expectedVersion

def test_plugin_exists_module():
    assert plugin_key in vamp.list_plugins()

def test_plugin_exists_in_freq_version():
    assert plugin_key_freq in vh.list_plugins()

def test_getoutputlist():
    outputs = vh.get_outputs_of(plugin_key)
    assert len(outputs) == 11
    assert "input-summary" in outputs

def test_getoutputlist_module():
    outputs = vamp.get_outputs_of(plugin_key)
    assert len(outputs) == 11
    assert "input-summary" in outputs

def test_getoutputlist_2():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    outputs = plug.get_outputs()
    assert len(outputs) == 11

def test_get_output_by_id():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    out = plug.get_output("input-summary")
    assert "sampleType" in out
    assert out["output_index"] == 9
    try:
        out = plug.get_output("chops")
        assert False
    except Exception:
        pass
    try:
        out = plug.get_output("")
        assert False
    except Exception:
        pass

def test_get_output_by_index():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    out = plug.get_output(0)
    assert "sampleType" in out
    assert out["identifier"] == "instants"
    assert out["output_index"] == 0
    try:
        out = plug.get_output(20)
        assert False
    except Exception:
        pass
    try:
        out = plug.get_output(-1)
        assert False
    except Exception:
        pass
    try:
        out = plug.get_output(plug)
        assert False
    except Exception:
        pass
    
def test_inputdomain():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    assert plug.inputDomain == vh.TIME_DOMAIN

def test_info():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    assert plug.info["identifier"] == "vamp-test-plugin"
    
def test_parameterdescriptors():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    assert plug.parameters[0]["identifier"] == "produce_output"

def test_getparameters_module():
    params = vamp.get_parameters_of(plugin_key)
    assert len(params) == 1
    assert params[0]["identifier"] == "produce_output"

def test_timestamp_method_fail():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    try:
        plug.set_process_timestamp_method(vh.SHIFT_DATA)
        assert False
    except Exception:
        pass

def test_timestamp_method_fail2():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_INPUT_DOMAIN)
    # Not a freq-domain plugin: shouldn't throw, but should return false
    assert (not plug.set_process_timestamp_method(vh.SHIFT_DATA))

def test_timestamp_method_succeed():
    plug = vh.load_plugin(plugin_key_freq, rate, vh.ADAPT_INPUT_DOMAIN)
    assert plug.set_process_timestamp_method(vh.SHIFT_DATA)
    
def test_setparameter():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    assert plug.parameters[0]["identifier"] == "produce_output"
    assert plug.parameters[0]["defaultValue"] == 1
    assert plug.get_parameter_value("produce_output") == plug.parameters[0]["defaultValue"]
    assert plug.set_parameter_value("produce_output", 0) == True
    assert plug.get_parameter_value("produce_output") == 0
    assert plug.set_parameter_values({ "produce_output": 1 }) == True
    assert plug.get_parameter_value("produce_output") == 1
    try:
        plug.set_parameter_value("produce_output", "fish")
        assert False
    except TypeError:
        pass
    try:
        plug.set_parameter_value(4, 0)
        assert False
    except TypeError:
        pass
    try:
        plug.set_parameter_value("steak", 0)
        assert False
    except Exception:
        pass
    try:
        plug.get_parameter_value(4)
        assert False
    except TypeError:
        pass
    try:
        plug.get_parameter_value("steak")
        assert False
    except Exception:
        pass
            
