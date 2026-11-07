"""Session headers for API tests whose subject is outside the login endpoint."""

from services.access_sessions import issue_session

USER_CREDENTIAL = "test_token_123"
ADMIN_CREDENTIAL = "admin_token_456"
USER_BEARER = "Bearer " + issue_session(USER_CREDENTIAL)[0]
ADMIN_BEARER = "Bearer " + issue_session(ADMIN_CREDENTIAL)[0]
