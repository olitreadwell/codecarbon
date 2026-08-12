"""

Based on https://kernelpanic.io/the-modern-way-to-call-apis-in-python
"""

import dataclasses
import json
import time
from datetime import datetime, timedelta, tzinfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from codecarbon.core.schemas import (
    EmissionCreate,
    ExperimentCreate,
    OrganizationCreate,
    ProjectCreate,
    RunCreate,
)
from codecarbon.external.logger import logger


def get_datetime_with_timezone():
    import arrow

    return str(arrow.now().isoformat())


def _measurement_timestamp(carbon_emission: dict) -> str:
    """
    Offset-aware ISO timestamp of *when the measurement was taken*, taken from
    EmissionsData.timestamp. Falls back to now for hand-built payloads that
    carry no usable timestamp.
    """
    try:
        return (
            datetime.fromisoformat(carbon_emission["timestamp"])
            .astimezone()
            .isoformat()
        )
    except (KeyError, TypeError, ValueError):
        return get_datetime_with_timezone()


# Failures where the request plausibly never reached the application, so a
# retry cannot duplicate work: no upstream was reachable (502/503) or we were
# told to slow down before being served (429).
_POST_SAFE_STATUSES = (429, 502, 503)
# Reads are idempotent, so a retry costs at most a wasted round trip.
_READ_SAFE_STATUSES = (429, 500, 502, 503, 504)


def _build_session(
    retries: int, backoff: float, statuses, retry_read: bool
) -> requests.Session:
    """
    A Session so sockets (and the TLS handshake) are reused across calls, with
    retry and exponential backoff on the failures that are worth retrying.

    :statuses: response codes to retry.
    :retry_read: whether to retry a read timeout or a truncated response, i.e.
        a failure that happened *after* the request reached the server.
    """
    retry_kwargs = {
        "total": retries,
        "connect": retries,
        # False, not 0: urllib3 then re-raises the original ReadTimeout instead
        # of burning the budget and reporting an exhausted-retries error.
        "read": retries if retry_read else False,
        "status": retries,
        "backoff_factor": backoff,
        "status_forcelist": statuses,
        "allowed_methods": frozenset({"GET", "POST", "PATCH", "PUT", "DELETE"}),
        "raise_on_status": False,
    }
    try:
        # Spreads the retry storm when a whole fleet reconnects after an outage.
        retry = Retry(backoff_jitter=1.0, **retry_kwargs)
    except TypeError:
        # urllib3 < 2.0 has no backoff_jitter. We are a library in other
        # people's environments, so degrade instead of pinning urllib3.
        retry = Retry(**retry_kwargs)
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=4)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class ApiClient:  # (AsyncClient)
    """
    This class call the Code Carbon API
    """

    run_id = None

    def __init__(
        self,
        endpoint_url="https://api.codecarbon.io",
        experiment_id=None,
        api_key=None,
        access_token=None,
        conf=None,
        create_run_automatically=True,
        timeout=(3.05, 10),
        retries=2,
        backoff=0.5,
    ):
        """
        :endpoint_url: URL of the API endpoint
        :experiment_id: ID of the experiment
        :api_key: Code Carbon API_KEY
        :access_token: Code Carbon API access token
        :conf: Metadata of the experiment
        :create_run_automatically: If False, do not create a run. To use API in read only mode.
        :timeout: requests timeout, either seconds or a (connect, read) tuple.
        :retries: number of retries after the first attempt.
        :backoff: backoff factor between retries, in seconds (0.5 -> 0.5s, 1s, 2s...).

        Note that `timeout` and `retries` multiply, so raise one only while
        looking at the other. Two different worst cases are worth keeping apart:

        - a *hung* endpoint, which accepts the connection and never answers:
          `read * (retries + 1)` plus backoff, since only the read times out.
        - the *full* retry chain, where the connection also has to time out:
          `(connect + read) * (retries + 1)` plus backoff.

        POSTs do not retry read timeouts (see `_post_session`), so their hung
        case is a single `connect + read` and the chain above is a GET bound.
        """
        self.url = endpoint_url
        self.experiment_id = experiment_id
        self.api_key = api_key
        self.conf = conf
        self.access_token = access_token
        self._timeout = timeout
        self._session = _build_session(
            retries, backoff, _READ_SAFE_STATUSES, retry_read=True
        )
        # POSTs create rows. carbonserver has no idempotency key and the
        # dashboard sums emission rows, so a POST replayed after the server
        # already committed the insert inflates a user's reported emissions
        # with nothing to show for it: a dropped row is visible, a duplicated
        # one is not. This session therefore only retries POSTs that
        # plausibly never reached the application -- connection errors,
        # connect timeouts, 429/502/503. Read timeouts, truncated responses,
        # 500 and 504 are *not* retried: the request landed, and the insert
        # may well have gone through. Widen this only once the API accepts an
        # idempotency key.
        self._post_session = _build_session(
            retries, backoff, _POST_SAFE_STATUSES, retry_read=False
        )
        # Run creation is the most expensive write path. When it fails, back off
        # instead of re-attempting on every measurement tick.
        self._create_run_not_before = 0.0
        self._create_run_backoff = 0.0
        if self.experiment_id is not None and create_run_automatically:
            self._create_run(self.experiment_id)

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            # set the x-api-token header
            headers["x-api-token"] = self.api_key
        elif self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    def _request(self, method, url, payload=None, expected_status=200):
        """
        Call the API and return the response, raising on anything that is not
        the status code the API answers on success.

        :method: the requests function to call, for example requests.get
        :payload: the JSON body to send, if any
        :expected_status: the http code the API returns when the call succeeds
        """
        headers = self._get_headers()
        response = method(url=url, json=payload, timeout=self._timeout, headers=headers)
        if response.status_code != expected_status:
            self._raise_api_error(url, payload or {}, response)
        return response

    def close(self):
        """Release the pooled sockets. Safe to call more than once."""
        self._session.close()
        self._post_session.close()

    def set_access_token(self, token: str):
        """This method sets the access token to be used for the API.
        Args:
            token (str): access token to be used for the API
        """
        self.access_token = token

    def check_auth(self):
        """
        Check API access to user account
        """
        url = self.url + "/auth/check"
        return self._request(self._session.get, url).json()

    def get_list_organizations(self):
        """
        List all organizations
        """
        url = self.url + "/organizations"
        return self._request(self._session.get, url).json()

    def check_organization_exists(self, organization_name: str):
        """
        Check if an organization exists
        """
        organizations = self.get_list_organizations()
        for organization in organizations:
            if organization["name"] == organization_name:
                return organization
        return False

    def create_organization(self, organization: OrganizationCreate):
        """
        Create an organization
        """
        payload = dataclasses.asdict(organization)
        url = self.url + "/organizations"
        if organization := self.check_organization_exists(organization.name):
            logger.warning(
                f"Organization {organization['name']} already exists. Skipping creation."
            )
            return organization
        else:
            return self._request(
                self._post_session.post, url, payload=payload, expected_status=201
            ).json()

    def get_organization(self, organization_id):
        """
        Get an organization
        """
        url = self.url + "/organizations/" + organization_id
        return self._request(self._session.get, url).json()

    def update_organization(self, organization: OrganizationCreate):
        """
        Update an organization
        """
        payload = dataclasses.asdict(organization)
        url = self.url + "/organizations/" + organization.id
        return self._request(self._session.patch, url, payload=payload).json()

    def list_projects_from_organization(self, organization_id):
        """
        List all projects
        """
        url = self.url + "/organizations/" + organization_id + "/projects"
        return self._request(self._session.get, url).json()

    def create_project(self, project: ProjectCreate):
        """
        Create a project
        """
        payload = dataclasses.asdict(project)
        url = self.url + "/projects"
        return self._request(
            self._post_session.post, url, payload=payload, expected_status=201
        ).json()

    def get_project(self, project_id):
        """
        Get a project
        """
        url = self.url + "/projects/" + project_id
        return self._request(self._session.get, url).json()

    def add_emission(self, carbon_emission: dict):
        assert self.experiment_id is not None
        if self.run_id is None:
            logger.warning(
                "ApiClient.add_emission() need a run_id : the initial call may "
                + "have failed. Retrying..."
            )
            self._create_run(self.experiment_id)
            if self.run_id is None:
                logger.error(
                    "ApiClient.add_emission still no run_id, aborting for this time !"
                )
            return False
        if carbon_emission["duration"] < 1:
            logger.warning(
                "ApiClient : emissions not sent because of a duration smaller than 1."
            )
            return False
        emission = EmissionCreate(
            timestamp=_measurement_timestamp(carbon_emission),
            run_id=self.run_id,
            duration=int(carbon_emission["duration"]),
            emissions_sum=carbon_emission["emissions"],
            emissions_rate=carbon_emission["emissions_rate"],
            cpu_power=carbon_emission["cpu_power"],
            gpu_power=carbon_emission["gpu_power"],
            ram_power=carbon_emission["ram_power"],
            cpu_energy=carbon_emission["cpu_energy"],
            gpu_energy=carbon_emission["gpu_energy"],
            ram_energy=carbon_emission["ram_energy"],
            energy_consumed=carbon_emission["energy_consumed"],
            cpu_utilization_percent=carbon_emission.get("cpu_utilization_percent"),
            gpu_utilization_percent=carbon_emission.get("gpu_utilization_percent"),
            ram_utilization_percent=carbon_emission.get("ram_utilization_percent"),
            wue=carbon_emission.get("wue", 0),
        )
        try:
            payload = dataclasses.asdict(emission)
            url = self.url + "/emissions"
            self._request(
                self._post_session.post, url, payload=payload, expected_status=201
            )
            logger.debug(f"ApiClient - Successful upload emission {payload} to {url}")
        except requests.exceptions.HTTPError:
            # Already logged by _raise_api_error, do not log it twice.
            raise
        except Exception as e:
            logger.error(e, exc_info=True)
            raise
        return True

    # Bounds on the run-creation retry delay, in seconds.
    _CREATE_RUN_BACKOFF_MIN = 30.0
    _CREATE_RUN_BACKOFF_MAX = 900.0

    def _create_run(self, experiment_id: str):
        """
        Create a run, backing off after a failure.

        Without the backoff every client in a fleet re-attempts run creation on
        every measurement tick for as long as the API is down, which hammers the
        most expensive write path exactly when it is least able to take it.
        Returns None without calling the API while the backoff is in effect.
        """
        if time.monotonic() < self._create_run_not_before:
            logger.debug(
                "ApiClient run creation is backing off after a previous failure, "
                "skipping this attempt."
            )
            return None
        try:
            run_id = self._create_run_once(experiment_id)
        except Exception:
            self._create_run_backoff = min(
                max(self._create_run_backoff * 2, self._CREATE_RUN_BACKOFF_MIN),
                self._CREATE_RUN_BACKOFF_MAX,
            )
            self._create_run_not_before = time.monotonic() + self._create_run_backoff
            raise
        self._create_run_backoff = 0.0
        self._create_run_not_before = 0.0
        return run_id

    def _create_run_once(self, experiment_id: str):
        """
        Create the experiment for project_id
        """
        if self.experiment_id is None:
            # TODO : raise an Exception ?
            logger.error(
                "ApiClient FATAL The ApiClient._create_run() needs an experiment_id !"
            )
            return None
        try:
            run = RunCreate(
                # "now" is correct here: a run's timestamp is its creation time,
                # unlike an emission's, which is its measurement time.
                timestamp=get_datetime_with_timezone(),
                experiment_id=experiment_id,
                os=self.conf.get("os"),
                python_version=self.conf.get("python_version"),
                codecarbon_version=self.conf.get("codecarbon_version"),
                cpu_count=self.conf.get("cpu_count"),
                cpu_model=self.conf.get("cpu_model"),
                gpu_count=self.conf.get("gpu_count"),
                gpu_model=self.conf.get("gpu_model"),
                # Reduce precision for Privacy
                longitude=round(self.conf.get("longitude", 0), 1),
                latitude=round(self.conf.get("latitude", 0), 1),
                region=self.conf.get("region"),
                provider=self.conf.get("provider"),
                ram_total_size=self.conf.get("ram_total_size"),
                tracking_mode=self.conf.get("tracking_mode"),
            )
            payload = dataclasses.asdict(run)
            url = self.url + "/runs"
            r = self._request(
                self._post_session.post, url, payload=payload, expected_status=201
            )
            self.run_id = r.json()["id"]
            logger.info(
                "ApiClient Successfully registered your run on the API.\n\n"
                + f"Run ID: {self.run_id}\n"
                + f"Experiment ID: {self.experiment_id}\n"
            )
            return self.run_id
        except requests.exceptions.ConnectionError as e:
            logger.error(
                f"Failed to connect to API, please check the configuration. {e}",
                exc_info=False,
            )
            raise
        except requests.exceptions.HTTPError:
            # Already logged by _raise_api_error, do not log it twice.
            raise
        except Exception as e:
            logger.error(e, exc_info=True)
            raise

    def list_experiments_from_project(self, project_id: str):
        """
        List all experiments for a project
        """
        url = self.url + "/projects/" + project_id + "/experiments"
        return self._request(self._session.get, url).json()

    def set_experiment(self, experiment_id: str):
        """
        Set the experiment id
        """
        self.experiment_id = experiment_id

    def add_experiment(self, experiment: ExperimentCreate):
        """
        Create an experiment, used by the CLI, not the package.
        ::experiment:: The experiment to create.
        """
        payload = dataclasses.asdict(experiment)
        url = self.url + "/experiments"
        return self._request(
            self._post_session.post, url, payload=payload, expected_status=201
        ).json()

    def get_experiment(self, experiment_id):
        """
        Get an experiment by id
        """
        url = self.url + "/experiments/" + experiment_id
        return self._request(self._session.get, url).json()

    def _raise_api_error(self, url, payload, response):
        """
        Log the failed call then always raise a requests.exceptions.HTTPError.
        """
        if len(payload) > 0:
            logger.error(
                f"ApiClient Error when calling the API on {url} with : {json.dumps(payload)}"
            )
        else:
            logger.error(f"ApiClient Error when calling the API on {url}")
        logger.error(
            f"ApiClient API return http code {response.status_code} and answer : {response.text}"
        )
        response.raise_for_status()
        # 2xx/3xx that still isn't what the caller expected
        raise requests.exceptions.HTTPError(
            f"Unexpected status {response.status_code} from {url}", response=response
        )

    def close_experiment(self):
        """
        Tell the API that the experiment has ended.
        """


class simple_utc(tzinfo):
    def tzname(self, **kwargs):
        return "UTC"

    def utcoffset(self, dt):
        return timedelta(0)
