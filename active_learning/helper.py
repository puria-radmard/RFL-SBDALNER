from active_learning.acquisition import RandomBaselineAcquisition, LowestConfidenceAcquisition, \
    MaximumEntropyAcquisition, BALDAcquisition
from active_learning.agent import ActiveLearningAgent
from active_learning.selector import FixedWindowSelector, SentenceSelector, VariableWindowSelector


def configure_al_agent(args, device, model, train_set, helper):

    round_size = int(args.roundsize)

    if len(args.window) == 1:
        if int(args.window[0]) == -1:
            if args.beam_search != 1:
                raise ValueError("Full sentence selection requires a beam search parameter of 1")
            selector = SentenceSelector(helper, normalisation_index=args.alpha, round_size=round_size)
        else:
            selector = FixedWindowSelector(
                helper, window_size=int(args.window[0]), beta=args.beta, round_size=round_size, model=model,
                beam_search_parameter=args.beam_search
            )
    elif len(args.window) == 2:
        selector = VariableWindowSelector(
            helper=helper, window_range=[int(a) for a in args.window], beta=args.beta, round_size=round_size,
            beam_search_parameter=args.beam_search, normalisation_index=args.alpha, model=model,
        )
    else:
        raise ValueError(f"Windows must be of one or two size, not {args.window}")

    if args.acquisition == 'baseline' and args.initprop != 1.0:
        raise ValueError("To run baseline, you must set initprop == 1.0")

    if args.acquisition in ['rand', 'baseline']:
        acquisition_class = RandomBaselineAcquisition(model)
    elif args.acquisition == 'lc':
        acquisition_class = LowestConfidenceAcquisition(model)
    elif args.acquisition == 'maxent':
        acquisition_class = MaximumEntropyAcquisition(model)
    elif args.acquisition == 'bald':
        acquisition_class = BALDAcquisition()
    else:
        raise ValueError(args.acquisition)

    agent = ActiveLearningAgent(
        train_set=train_set,
        acquisition_class=acquisition_class,
        selector_class=selector,
        round_size=round_size,
        batch_size=args.batch_size,
        helper=helper,
        device=device,
        propagation_mode=args.propagation_mode
    )

    return agent
