import gymnasium as gym
import numpy as np
import time

#Constansts
G,M,L,DT=10.0,1.0,1.0,0.05
MAX_SPEED,MAX_TORQUE=8.0,2.0
H,N,K,I,sig0=30,500,50,5,1.0
rng=np.random.default_rng(0)


def angle_normalize(x):
    return np.arctan2(np.sin(x),np.cos(x))

def step(th,thdot,u):
    u=np.clip(u,-MAX_TORQUE,MAX_TORQUE)
    normalized_th=angle_normalize(th)
    cost=normalized_th**2+0.1*thdot**2+0.001*u**2
    thdot_new=thdot+(3*G/(2*L)*np.sin(th)+3/(M*L**2)*u)*DT
    thdot_new=np.clip(thdot_new,-MAX_SPEED,MAX_SPEED)
    th_new=th+thdot_new*DT
    r=-cost
    return th_new,thdot_new,r

#after break
def check_step(env,n=1000):
    th=rng.uniform(-2*np.pi,2*np.pi,size=n)
    thdot=rng.uniform(-MAX_SPEED,MAX_SPEED,size=n)
    u=rng.uniform(-2.5,2.5,size=n)
    th_mine,thdot_mine,r_mine=step(th,thdot,u)
    env.reset()
    th_gym=[]
    thdot_gym=[]
    r_gym=[]
    for i in range(n):
        env.unwrapped.state=np.array([th[i],thdot[i]])
        _,r_g,_,_,_=env.unwrapped.step(np.array([u[i]]))
        th_g,thdot_g=env.unwrapped.state
        th_gym.append(th_g)
        thdot_gym.append(thdot_g)
        r_gym.append(r_g)
    th_gym=np.array(th_gym)
    thdot_gym=np.array(thdot_gym)
    r_gym=np.array(r_gym)
    diff1=np.abs(th_mine-th_gym)
    diff2=np.abs(thdot_mine-thdot_gym)
    diff3=np.abs(r_gym-r_mine)
    print("max diff θ:", diff1.max())
    print("max diff θ dot :", diff2.max())
    print("max diff r:", diff3.max())
    assert (diff1.max() < 1e-6 and diff2.max() < 1e-6 and diff3.max() < 1e-6)

def rollout(th,thdot,X):      #Shape of X is (P,H)
    P=X.shape[0]
    J=np.zeros(P)
    for k in range(X.shape[1]):
        th_new,thdot_new,r=step(th,thdot,X[:,k])
        th=th_new
        thdot=thdot_new
        J+=r
    return J

def check_rollout(env):
    th0=rng.uniform(-np.pi,np.pi)
    thdot0=rng.uniform(-1,1)
    plan=rng.uniform(-MAX_TORQUE,MAX_TORQUE,size=(1,H))
    J_mine=rollout(th0,thdot0,plan)
    env.reset()
    env.unwrapped.state=np.array([th0,thdot0])
    J_gym=0

    for k in range(H):
        _,r_g,_,_,_=env.unwrapped.step(np.array([plan[0,k]]))
        J_gym+=r_g
    diff=np.abs(J_mine[0]-J_gym)
    print("Rollout Diff:",diff)
    assert diff < 1e-6

def cem(th,thdot,mu):
    sig=np.full(H,sig0)
    for i in range(I):
        noise=rng.standard_normal(size=(N,H))
        X=mu+sig*noise
        X=np.clip(X,-MAX_TORQUE,MAX_TORQUE)
        J=rollout(th,thdot,X)
        idx=np.argsort(J)[-K:]
        elites=X[idx]
        mu,sig=np.mean(elites,axis=0),np.std(elites,axis=0)
    return mu,sig

def run_episode(env,seed):
    env.reset(seed=seed)
    mu=np.zeros(H)
    total=0
    times=[]
    log_every=20

    for t in range(200):
        th,thdot=env.unwrapped.state
        t0=time.perf_counter()
        mu,_=cem(th,thdot,mu)
        times.append((time.perf_counter()-t0)*1000)
        _,r,terminated,truncated,_=env.step(np.array([mu[0]]))
        total+=r
        mu=np.concatenate([mu[1:],[0.0]])
        if t%log_every==0:
            print(f"t {t:3d} | th {angle_normalize(th):+.2f} | thdot {thdot:+.2f} | total {total:8.1f}")

        if terminated or truncated:
            break

    return total,np.mean(times)






if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    check_step(env)
    check_rollout(env)

    for h in [5, 10, 20, 30, 50]:          
        H = h                             
        returns, times = [], []
        for seed in range(5):             
            total, ms = run_episode(env, seed)
            returns.append(total)
            times.append(ms)
        print(f"H {H:2d} | " + " ".join(f"{r:7.1f}" for r in returns)
              + f" | mean {np.mean(returns):7.1f} | {np.mean(times):5.1f} ms")



