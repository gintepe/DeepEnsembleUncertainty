import methods.general_loops

# TODO fix the passing for count!!!
def test(model, test_loader, metric_dict, device):
    return methods.general_loops.test(model, test_loader, metric_dict, device, is_single_output=False, pred=lambda m, x: m.mc_predict(x, 5)[0])
