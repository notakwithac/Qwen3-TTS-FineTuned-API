"""
API Wrapper for Massed Compute VM API (v1).
Documentation: https://vm-docs.massedcompute.com/api/v1
"""

import os
import requests
from typing import List, Dict, Any, Optional


class MassedComputeClient:

    BASE_URL = "https://api.massedcompute.com/api/v1"

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv("MASSED_COMPUTE_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "MASSED_COMPUTE_API_TOKEN not found in environment or provided to client."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def _get(self, endpoint: str) -> Dict[str, Any]:
        response = requests.get(
            f"{self.BASE_URL}/{endpoint}", headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.BASE_URL}/{endpoint}", headers=self.headers, json=payload
        )
        response.raise_for_status()
        return response.json()

    def authenticate(self) -> bool:
        """Validates the API token."""
        try:
            result = self._post("account/token/validation", {})
            return result.get("message") == "Valid Token"
        except requests.RequestException:
            return False

    def list_gpus(self) -> Dict[str, Any]:
        """Retrieves all GPU types, configurations, and available inventory."""
        return self._get("gpu-inventory")

    def list_images(self) -> List[Dict[str, Any]]:
        """Retrieve list of available images."""
        return self._get("images")

    def list_instances(self) -> List[Dict[str, Any]]:
        """Retrieve list of all running instances."""
        return self._get("instance")

    def launch_instance(
        self,
        image_id: int,
        product_name: str,
        region_name: str,
        instance_name: str,
        ssh_keys: List[str],
        coupon: Optional[str] = None,
        command: Optional[str] = None,
    ) -> str:
        """Deploys a new instance.  Returns the instance UUID."""
        payload = {
            "imageId": image_id,
            "productName": product_name,
            "regionName": region_name,
            "instanceName": instance_name,
            "sshKeys": ssh_keys,
        }
        if coupon:
            payload["coupon"] = coupon
        if command:
            payload["command"] = command

        result = self._post("instance/launch", payload)
        return result.get("response")

    def terminate_instance(self, instance_uuids: List[str]) -> Dict[str, Any]:
        """Terminate one or more instances."""
        payload = {"instanceUuids": instance_uuids}
        return self._post("instance/terminate", payload)

    def get_instance(self, instance_uuid: str) -> Dict[str, Any]:
        """Retrieve details of a single running instance by UUID."""
        return self._get(f"instance/{instance_uuid}")

    def restart_instance(self, instance_uuids: List[str]) -> Dict[str, Any]:
        """Restart one or more instances."""
        payload = {"instanceUuids": instance_uuids}
        return self._post("instance/restart", payload)
