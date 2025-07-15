@echo off
echo Deploying Firebase configuration for Pawnder
echo Project ID: pawnder-457917

echo Setting Firebase project...
firebase use pawnder-457917

echo Deploying Firestore rules...
firebase deploy --only firestore:rules

echo Deploying Firestore indexes...
firebase deploy --only firestore:indexes

echo Deploying Storage rules...
firebase deploy --only storage

echo Firebase deployment complete!
echo Firebase Console: https://console.firebase.google.com/project/pawnder-457917
pause
