"""The user's complete clean + six practical test scope."""
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')


def require_seven_views(done):
    if (done.get('status') != 'PREDICTIONS_COMPLETE' or tuple(done.get('views', ())) != VIEWS
            or done.get('truth_read') is not False or done.get('query_fit') is not False
            or done.get('all_parameters_frozen') is not True or done.get('head_fit_updates') != 0):
        raise ValueError('All seven frozen predictions must precede truth scoring')
