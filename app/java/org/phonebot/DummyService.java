package org.phonebot;

import android.app.Service;
import android.content.Intent;
import android.content.Context;
import android.content.ServiceConnection;
import android.content.ComponentName;
import android.os.Binder;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.widget.Toast;

public class DummyService extends Service {
    public static class DummyBinder extends Binder{
        private DummyService mService = null;
        public DummyBinder(DummyService service){
            mService = service;
        }
        DummyService getService(){
            return mService;
        }
    }

    public static class DummyConnection implements ServiceConnection{
        private DummyService mService=null;

        DummyService getService(){
            return mService;
        }

        public void onServiceConnected(ComponentName className, IBinder service) {
            mService = ((DummyService.DummyBinder)service).getService();
        }

        public void onServiceDisconnected(ComponentName className) {
            mService = null;
        }
    }

    // Members
    private final IBinder mBinder = new DummyBinder(this);
    // NOTE(yycho0108): For this to work, the service must run on the same process as the application.
    // (Or something along those lines).
    private final Handler mHandler = new Handler(Looper.getMainLooper());

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        return Service.START_NOT_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return mBinder;
    }

    // Currently, sendData is a dummy replacement for doing actual stuff
    public void sendData(final String data){
        // NOTE(yycho0108): Toast must be run on the gui thread.
        mHandler.post(new Runnable(){
            @Override
            public void run(){
                Toast.makeText(getApplicationContext(), data, Toast.LENGTH_SHORT).show();
            }
        });
    }

    // Create the connection object and return.
    public static DummyConnection bindToContext(Context context){
        Intent intent = new Intent(context, DummyService.class);
        DummyConnection connection = new DummyConnection();
        context.bindService(intent, connection, Context.BIND_AUTO_CREATE);
        return connection;
    }
}

