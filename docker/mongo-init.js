const appDbName = process.env.MONGO_APP_DB || 'energy_fault_detector';
const appUser = process.env.MONGO_APP_USER || 'efd_app';
const appPassword = process.env.MONGO_APP_PASSWORD || 'efd_app_password';

const adminDb = db.getSiblingDB('admin');
const targetDb = db.getSiblingDB(appDbName);

if (!adminDb.getUser(appUser)) {
  print(`Creating MongoDB application user ${appUser} for database ${appDbName}.`);
  targetDb.createUser({
    user: appUser,
    pwd: appPassword,
    roles: [
      { role: 'readWrite', db: appDbName }
    ]
  });
} else {
  print(`MongoDB application user ${appUser} already exists; skipping creation.`);
}
