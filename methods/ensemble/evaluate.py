import methods.general_loops

def test(model, test_loader, metric_dict, device):
    return methods.general_loops.test(model, test_loader, metric_dict, device, is_single_output=False, pred=lambda m, x: m(x)[0])