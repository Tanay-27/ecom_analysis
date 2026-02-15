# Multi-Tenancy & Authentication Requirements

## Overview
The system must support multiple independent clients (tenants), ensuring complete data isolation and customized business logic for each.

## Requirements
- **Tenant Isolation**: Data for one client must never be visible to another.
- **Tenant Configuration**: Each tenant can define their own lead times, godowns, and forecasting weights.
- **User Roles**: Support for different user levels (e.g., Admin, Viewer, Operations).

## Pending Questions
1. **Isolation Level**: Should we use a single database with a `TenantID` on every table, or separate databases/schemas per client?
2. **User Onboarding**: Will users be added manually by a Super Admin, or is there a self-signup flow?
3. **Authentication**: Do we need Social Logins (Google/Microsoft), or is standard Email/Password sufficient?
4. **Role Permissions**: Can you list the specific actions an "Operations User" can take vs an "Inventory Manager"?
5. **Tenant Branding**: Does the UI need to change (logos/colors) based on the logged-in tenant?

## Answers
1. If we use POSTGRESQL, we can use schema isolation. If we use MSSQL server, we can use database isolation.
2. Manually adding users by super admin.
3. Standard Email/Password authentication.
4. Admin has complete access, Owner has complete access. Lets focus on admin and owner for now with provisions to add other roles later.
5. Tenant branding is not required.

