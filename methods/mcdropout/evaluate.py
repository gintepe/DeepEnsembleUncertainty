import methods.general_loops

def test_wrapper(n):
    return lambda model, test_loader, metric_dict, device: test(model, test_loader, metric_dict, device, n)

def test(model, test_loader, metric_dict, device, n):
    return methods.general_loops.test(model, test_loader, metric_dict, device, is_single_output=False, pred=lambda m, x: m.mc_predict(x, 5)[0])
