#!/usr/bin/env python3
"""One-time interactive Globus login for LOCAL-mode transfers.

Run this once from a terminal:

    source setup.sh
    python3 main/globus_login.py main/workflow_local.json [collection-uuid ...]

It performs the same login the workflow would trigger on its first transfer,
and caches refreshable tokens at `workflow.globus_token_path`, after which the
workflow runs non-interactively.

Named endpoints known to the IRI backend (dtn, perlmutter) are consented to by
default. Any other collection - e.g. SDCC - needs a `data_access` consent; pass
its UUID (or a name like `sdcc`) as an extra argument to grant that here rather
than having a transfer stop mid-run to ask for it.
"""
import sys
from femtomeas.workflow_manager.manager_config import readManagerConfigFile
from femtomeas.workflow_manager import local_api

#Collections worth naming, so callers don't have to paste UUIDs.
KNOWN_COLLECTIONS = {
    "sdcc": "12782fb1-a599-4f18-b0fb-2e849681e214",
}

def main():
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "main/workflow_local.json"
    extra = [KNOWN_COLLECTIONS.get(a.lower(), a) for a in sys.argv[2:]]
    config = readManagerConfigFile(cfg_file)

    endpoint = config.workflow.local_globus_endpoint
    if not endpoint:
        sys.exit(f"workflow.local_globus_endpoint is not set in {cfg_file}")

    local_api.setupGlobus(endpoint, config.workflow.globus_token_path)

    scopes = ()
    if extra:
        #data_access must be requested as a *dependent* scope of transfer:all -
        #the same form the Transfer API hands back in `required_scopes` when it
        #raises ConsentRequired. A bare data_access scope grants a consent the
        #Transfer API does not look at, and the operation still 403s.
        scopes = tuple(f"urn:globus:auth:scope:transfer.api.globus.org:all"
                       f"[*https://auth.globus.org/scopes/{uuid}/data_access]"
                       for uuid in extra)
        print(f"Also requesting data_access consent for: {', '.join(extra)}")

    #force_login_scopes always re-runs the interactive login, so only use it when
    #extra consents are actually being added.
    client = local_api._transferClient(force_login_scopes=scopes if scopes else None)

    me = client.get_endpoint(endpoint)
    print(f"\nLogin OK. Local collection resolves as:")
    print(f"  display_name : {me['display_name']}")
    print(f"  id           : {me['id']}")
    print(f"  owner        : {me.get('owner_string')}")

    for uuid in extra:
        try:
            ep = client.get_endpoint(uuid)
            print(f"\nConsented collection {uuid}:")
            print(f"  display_name : {ep['display_name']}")
        except Exception as e:
            print(f"\nCould not query {uuid}: {e}")

    print(f"\nTokens cached at {local_api._globus['token_path']}")

if __name__ == "__main__":
    main()
