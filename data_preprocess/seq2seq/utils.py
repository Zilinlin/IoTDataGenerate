
import numpy as np

def softmax(x):
    """
    Compute the softmax of vector x.
    """
    # Subtract the maximum value of x from each element of x to avoid overflow
    # and make the softmax numerically stable.
    z = x - np.max(x)
    
    # Calculate the exponential of each element in z.
    exp_z = np.exp(z)
    
    # Calculate the sum of the exponential values.
    sum_exp_z = np.sum(exp_z)
    
    # Divide each element of the exponential vector by the sum of exponential values.
    softmax_z = exp_z / sum_exp_z
    
    return softmax_z


def print_header(string):
    bar_len = len(string)
    bar = '-'*bar_len
    print(bar+ '\n' + string + '\n' + bar)


def get_events(ps_attack, ps_recon, ps_infec, windows):

    events_per_detector = []
    for detector in  [ps_attack, ps_recon, ps_infec]:
        #examples_train = infection_unb_data['train'][0]
        events = detector.predict(windows, kind='')
        print('proportion of infection events tagged by {} detector: '.format(detector.get_name()), np.sum(np.array(events)>0.5)/len(events))
        events_per_detector.append(events)
    benign_events = []
    no_events = len(events_per_detector[0])
    for i  in range(no_events):
        not_attack = 1-events_per_detector[0][i]
        not_infection = 1-events_per_detector[1][i]
        not_recon = 1-events_per_detector[2][i]
        benign_events.append(not_attack * not_infection * not_recon)
    events_per_detector.append(benign_events)
    print('proportion of events estimated to be benign : ', np.sum(np.array(benign_events)>0.5)/len(benign_events))
    events_per_detector = np.array(events_per_detector).squeeze()
    events = np.transpose(events_per_detector)
    events_softmax = [softmax(e) for e in events]
    
    return events_softmax