"""Dependency-free policy shared by the scheduler and GPU training."""
def normal_region_exclusion(region):
    """A removed/moved/retyped box is not evidence of a normal structure.

    Only an explicit local-normal action can supply negative supervision. The
    two legacy structure actions were explicit even before provenance existed;
    generic legacy ``false_positive`` was also produced by ordinary deletion.
    """
    if not isinstance(region, dict):
        return 'invalid_normal_region'
    reason = region.get('reason')
    provenance = region.get('normal_provenance')
    if provenance is not None:
        actions = {'process_hole': 'process_hole', 'normal_structure': 'normal_structure',
                   'false_positive': 'confirmed_false_positive'}
        if (not isinstance(provenance, dict) or provenance.get('source') != 'human_explicit_normal'
                or provenance.get('confirmed') is not True
                or provenance.get('action') != actions.get(reason)
                or reason not in actions):
            return 'invalid_explicit_normal_provenance'
        return ''
    if reason in ('process_hole', 'normal_structure'):
        return ''
    if reason == 'false_positive':
        return 'legacy_false_positive_without_explicit_normal_confirmation'
    return 'normal_region_not_explicitly_confirmed'

